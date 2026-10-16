/* CPython extension module wll._wll: the encoder and decoder on
 * buffer-protocol pixels.
 *
 * Encoder.encode() takes a 3-D (height, width, 4) uint8 buffer whose pixels
 * are 4 bytes apart (rows may be further apart) and returns a read-only
 * memoryview of the encoded file, without copying it. Decoder.decode_into()
 * decodes into such a (writable) buffer. The GIL is released while encoding
 * and decoding. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

#include "wll.h"

static PyObject* UnsupportedError;

/* Marks an encoder or decoder in use; returns 0 if it already is. Atomic,
 * so that it also holds without the GIL (free-threaded builds). */
static int acquire(int* busy) {
  return __atomic_exchange_n(busy, 1, __ATOMIC_ACQUIRE) == 0;
}

static void release(int* busy) { __atomic_store_n(busy, 0, __ATOMIC_RELEASE); }

/* Raises the exception for a failed decode. */
static PyObject* raise_decode_error(wll_status st) {
  switch (st) {
    case WLL_ERR_OUT_OF_MEMORY:
      return PyErr_NoMemory();
    case WLL_ERR_UNSUPPORTED:
      PyErr_SetString(UnsupportedError,
                      "lossy or animated WebP files are not supported");
      return NULL;
    case WLL_ERR_INVALID_DATA:
      PyErr_SetString(PyExc_ValueError, "corrupt or truncated WebP data");
      return NULL;
    case WLL_ERR_INVALID_ARG: /* dimensions other than `out`'s */
      PyErr_SetString(PyExc_ValueError,
                      "the WebP data changed while being decoded");
      return NULL;
    default:
      PyErr_Format(PyExc_ValueError, "decoding failed (status %d)", (int)st);
      return NULL;
  }
}

/* Owns an encoded file for the memoryviews returned by encode(). */
typedef struct {
  PyObject_HEAD
  uint8_t* data;
  size_t size;
} OutputObject;

static void output_dealloc(OutputObject* self) {
  wll_free(self->data);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static int output_getbuffer(OutputObject* self, Py_buffer* view, int flags) {
  return PyBuffer_FillInfo(view, (PyObject*)self, self->data,
                           (Py_ssize_t)self->size, 1, flags);
}

static PyBufferProcs output_as_buffer = {
    (getbufferproc)output_getbuffer,
    NULL,
};

static PyTypeObject OutputType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "wll._wll.Output",
    .tp_basicsize = sizeof(OutputObject),
    .tp_dealloc = (destructor)output_dealloc,
    .tp_as_buffer = &output_as_buffer,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Encoded WebP file (exposed through memoryviews).",
};

typedef struct {
  PyObject_HEAD
  wll_encoder* enc;
  int busy; /* set while a thread encodes without the GIL */
} EncoderObject;

static PyObject* encoder_new(PyTypeObject* type, PyObject* args,
                             PyObject* kwds) {
  static char* kwlist[] = {"level", NULL};
  int level = WLL_LEVEL_DEFAULT;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i:Encoder", kwlist,
                                   &level)) {
    return NULL;
  }
  EncoderObject* self = (EncoderObject*)type->tp_alloc(type, 0);
  if (self == NULL) return NULL;
  self->enc = wll_encoder_new();
  if (self->enc == NULL) {
    Py_DECREF(self);
    return PyErr_NoMemory();
  }
  if (wll_encoder_set_level(self->enc, level) != WLL_OK) {
    Py_DECREF(self);
    PyErr_Format(PyExc_ValueError, "invalid level %d", level);
    return NULL;
  }
  return (PyObject*)self;
}

static void encoder_dealloc(EncoderObject* self) {
  wll_encoder_free(self->enc);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/* Whether a buffer format string describes unsigned bytes. */
static int is_uint8_format(const char* format) {
  if (format == NULL) return 1;
  if (format[0] != '\0' && strchr("@=<>!", format[0]) != NULL) ++format;
  return strcmp(format, "B") == 0;
}

static int check_size(Py_ssize_t width, Py_ssize_t height) {
  if (width < 1 || height < 1 || width > WLL_MAX_DIMENSION ||
      height > WLL_MAX_DIMENSION) {
    PyErr_Format(PyExc_ValueError,
                 "image size %zdx%zd is outside 1x1 to %dx%d", width, height,
                 WLL_MAX_DIMENSION, WLL_MAX_DIMENSION);
    return 0;
  }
  return 1;
}

/* Encodes pixels whose memory stays valid during the call (the caller holds
 * their owner) and returns a memoryview of the file. */
static PyObject* encode_pixels(EncoderObject* self, const uint8_t* pixels,
                               int width, int height, size_t stride) {
  if (!acquire(&self->busy)) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Encoder is in use by another thread; use one per thread");
    return NULL;
  }
  uint8_t* out = NULL;
  size_t out_size = 0;
  wll_status st;
  Py_BEGIN_ALLOW_THREADS
  st = wll_encoder_encode(self->enc, pixels, width, height, stride, &out,
                          &out_size);
  Py_END_ALLOW_THREADS
  release(&self->busy);
  if (st == WLL_ERR_OUT_OF_MEMORY) return PyErr_NoMemory();
  if (st != WLL_OK) {
    PyErr_Format(PyExc_ValueError, "encoding failed (status %d)", (int)st);
    return NULL;
  }
  OutputObject* o = PyObject_New(OutputObject, &OutputType);
  if (o == NULL) {
    wll_free(out);
    return NULL;
  }
  o->data = out;
  o->size = out_size;
  PyObject* mv = PyMemoryView_FromObject((PyObject*)o);
  Py_DECREF(o);
  return mv;
}

static PyObject* encoder_encode(EncoderObject* self, PyObject* pixels) {
  Py_buffer view;
  if (PyObject_GetBuffer(pixels, &view, PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
    return NULL;
  }
  if (view.ndim != 3 || view.shape[2] != 4 || view.itemsize != 1 ||
      !is_uint8_format(view.format)) {
    PyBuffer_Release(&view);
    PyErr_SetString(PyExc_ValueError,
                    "pixels must be a (height, width, 4) uint8 array");
    return NULL;
  }
  const Py_ssize_t height = view.shape[0], width = view.shape[1];
  if (!check_size(width, height)) {
    PyBuffer_Release(&view);
    return NULL;
  }
  if (view.strides[2] != 1 || view.strides[1] != 4 ||
      view.strides[0] < 4 * width) {
    PyBuffer_Release(&view);
    PyErr_SetString(PyExc_ValueError,
                    "pixels must be 4 bytes apart and rows ascending in "
                    "memory (use numpy.ascontiguousarray)");
    return NULL;
  }
  PyObject* result =
      encode_pixels(self, (const uint8_t*)view.buf, (int)width, (int)height,
                    (size_t)view.strides[0]);
  PyBuffer_Release(&view);
  return result;
}

/* The Arrow C data interface (ABI-stable structures from the Arrow
 * specification). */
#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE
struct ArrowSchema {
  const char* format;
  const char* name;
  const char* metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema** children;
  struct ArrowSchema* dictionary;
  void (*release)(struct ArrowSchema*);
  void* private_data;
};

struct ArrowArray {
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void** buffers;
  struct ArrowArray** children;
  struct ArrowArray* dictionary;
  void (*release)(struct ArrowArray*);
  void* private_data;
};
#endif

/* Whether an Arrow array has no nulls: a null count of 0, or unknown (-1)
 * without a validity bitmap. */
static int arrow_no_nulls(const struct ArrowArray* a) {
  if (a->null_count == 0) return 1;
  return a->null_count == -1 &&
         (a->n_buffers < 1 || a->buffers == NULL || a->buffers[0] == NULL);
}

/* encode_arrow(schema, array, width, height): encodes an Arrow array of
 * width * height fixed-size lists of 4 uint8 (RGBA pixels in reading
 * order), given as the PyCapsules of __arrow_c_array__(). The capsules keep
 * owning the array. */
static PyObject* encoder_encode_arrow(EncoderObject* self, PyObject* args) {
  PyObject *schema_capsule, *array_capsule;
  Py_ssize_t width, height;
  if (!PyArg_ParseTuple(args, "OOnn:encode_arrow", &schema_capsule,
                        &array_capsule, &width, &height)) {
    return NULL;
  }
  const struct ArrowSchema* schema =
      PyCapsule_GetPointer(schema_capsule, "arrow_schema");
  if (schema == NULL) return NULL;
  const struct ArrowArray* array =
      PyCapsule_GetPointer(array_capsule, "arrow_array");
  if (array == NULL) return NULL;
  if (!check_size(width, height)) return NULL;
  const int64_t n = (int64_t)width * height;
  const struct ArrowArray* child =
      array->n_children == 1 && array->children != NULL ? array->children[0]
                                                        : NULL;
  /* n <= MAX_DIMENSION squared, so the offset bound keeps the length
   * computation from overflowing. */
  if (schema->release == NULL || array->release == NULL ||
      schema->format == NULL || strcmp(schema->format, "+w:4") != 0 ||
      schema->n_children != 1 || schema->children == NULL ||
      schema->children[0] == NULL || schema->children[0]->format == NULL ||
      strcmp(schema->children[0]->format, "C") != 0 || array->length != n ||
      !arrow_no_nulls(array) || array->offset < 0 ||
      array->offset > INT64_MAX / 4 - n || child == NULL ||
      !arrow_no_nulls(child) || child->offset < 0 || child->n_buffers != 2 ||
      child->buffers == NULL || child->buffers[1] == NULL ||
      child->length < (array->offset + n) * 4 ||
      child->offset > INT64_MAX - child->length) {
    PyErr_SetString(PyExc_ValueError,
                    "Arrow array must hold width * height fixed-size lists "
                    "of 4 uint8 without nulls");
    return NULL;
  }
  const uint8_t* pixels = (const uint8_t*)child->buffers[1] + child->offset +
                          array->offset * 4;
  return encode_pixels(self, pixels, (int)width, (int)height,
                       (size_t)width * 4);
}

static PyMethodDef encoder_methods[] = {
    {"encode", (PyCFunction)encoder_encode, METH_O,
     "encode(pixels) -> memoryview\n\n"
     "Encodes a (height, width, 4) uint8 RGBA buffer as a lossless WebP "
     "file."},
    {"encode_arrow", (PyCFunction)encoder_encode_arrow, METH_VARARGS,
     "encode_arrow(schema, array, width, height) -> memoryview\n\n"
     "Encodes RGBA pixels given as Arrow PyCapsules (__arrow_c_array__())."},
    {NULL, NULL, 0, NULL},
};

static PyTypeObject EncoderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "wll._wll.Encoder",
    .tp_basicsize = sizeof(EncoderObject),
    .tp_dealloc = (destructor)encoder_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Encoder(level=DEFAULT)\n\n"
              "Keeps work buffers between encodes. Not for concurrent use.",
    .tp_methods = encoder_methods,
    .tp_new = encoder_new,
};

typedef struct {
  PyObject_HEAD
  wll_decoder* dec;
  int busy; /* set while a thread decodes without the GIL */
} DecoderObject;

static PyObject* decoder_new(PyTypeObject* type, PyObject* args,
                             PyObject* kwds) {
  if (!PyArg_ParseTuple(args, ":Decoder") ||
      (kwds != NULL && PyDict_Size(kwds) > 0)) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_TypeError, "Decoder() takes no arguments");
    }
    return NULL;
  }
  DecoderObject* self = (DecoderObject*)type->tp_alloc(type, 0);
  if (self == NULL) return NULL;
  self->dec = wll_decoder_new();
  if (self->dec == NULL) {
    Py_DECREF(self);
    return PyErr_NoMemory();
  }
  return (PyObject*)self;
}

static void decoder_dealloc(DecoderObject* self) {
  wll_decoder_free(self->dec);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* decoder_decode_into(DecoderObject* self, PyObject* args) {
  Py_buffer data;
  PyObject* out;
  if (!PyArg_ParseTuple(args, "y*O:decode_into", &data, &out)) return NULL;
  int width, height;
  wll_status st = wll_get_info((const uint8_t*)data.buf, (size_t)data.len,
                               &width, &height);
  if (st != WLL_OK) {
    PyBuffer_Release(&data);
    return raise_decode_error(st);
  }
  Py_buffer view;
  if (PyObject_GetBuffer(out, &view,
                         PyBUF_WRITABLE | PyBUF_STRIDES | PyBUF_FORMAT) < 0) {
    PyBuffer_Release(&data);
    return NULL;
  }
  if (view.ndim != 3 || view.shape[0] != height || view.shape[1] != width ||
      view.shape[2] != 4 || view.itemsize != 1 ||
      !is_uint8_format(view.format) || view.strides[2] != 1 ||
      view.strides[1] != 4 || view.strides[0] < 4 * (Py_ssize_t)width) {
    PyBuffer_Release(&view);
    PyBuffer_Release(&data);
    PyErr_Format(PyExc_ValueError,
                 "out must be a writable (%d, %d, 4) uint8 array with pixels "
                 "4 bytes apart and rows ascending in memory",
                 height, width);
    return NULL;
  }
  if (!acquire(&self->busy)) {
    PyBuffer_Release(&view);
    PyBuffer_Release(&data);
    PyErr_SetString(PyExc_RuntimeError,
                    "Decoder is in use by another thread; use one per thread");
    return NULL;
  }
  Py_BEGIN_ALLOW_THREADS
  st = wll_decoder_decode(self->dec, (const uint8_t*)data.buf,
                          (size_t)data.len, (uint8_t*)view.buf, width, height,
                          (size_t)view.strides[0]);
  Py_END_ALLOW_THREADS
  release(&self->busy);
  PyBuffer_Release(&view);
  PyBuffer_Release(&data);
  if (st != WLL_OK) return raise_decode_error(st);
  Py_RETURN_NONE;
}

static PyMethodDef decoder_methods[] = {
    {"decode_into", (PyCFunction)decoder_decode_into, METH_VARARGS,
     "decode_into(data, out)\n\n"
     "Decodes a WebP file into `out`, a writable (height, width, 4) uint8 "
     "buffer."},
    {NULL, NULL, 0, NULL},
};

static PyTypeObject DecoderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "wll._wll.Decoder",
    .tp_basicsize = sizeof(DecoderObject),
    .tp_dealloc = (destructor)decoder_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Decoder()\n\n"
              "Keeps work buffers between decodes. Not for concurrent use.",
    .tp_methods = decoder_methods,
    .tp_new = decoder_new,
};

static PyObject* module_info(PyObject* self, PyObject* args) {
  (void)self;
  Py_buffer data;
  if (!PyArg_ParseTuple(args, "y*:info", &data)) return NULL;
  int width, height;
  const wll_status st = wll_get_info((const uint8_t*)data.buf,
                                     (size_t)data.len, &width, &height);
  PyBuffer_Release(&data);
  if (st != WLL_OK) return raise_decode_error(st);
  return Py_BuildValue("(ii)", width, height);
}

static PyMethodDef module_methods[] = {
    {"info", module_info, METH_VARARGS,
     "info(data) -> (width, height)\n\nThe dimensions of a WebP file."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "wll._wll",
    .m_doc = "Exact lossless WebP encoder and decoder.",
    .m_size = -1,
    .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit__wll(void) {
  if (PyType_Ready(&OutputType) < 0 || PyType_Ready(&EncoderType) < 0 ||
      PyType_Ready(&DecoderType) < 0) {
    return NULL;
  }
  PyObject* m = PyModule_Create(&module);
  if (m == NULL) return NULL;
  Py_INCREF(&EncoderType);
  if (PyModule_AddObject(m, "Encoder", (PyObject*)&EncoderType) < 0) {
    Py_DECREF(&EncoderType);
    Py_DECREF(m);
    return NULL;
  }
  Py_INCREF(&DecoderType);
  if (PyModule_AddObject(m, "Decoder", (PyObject*)&DecoderType) < 0) {
    Py_DECREF(&DecoderType);
    Py_DECREF(m);
    return NULL;
  }
  UnsupportedError = PyErr_NewException("wll.UnsupportedError",
                                        PyExc_ValueError, NULL);
  if (UnsupportedError == NULL) {
    Py_DECREF(m);
    return NULL;
  }
  Py_INCREF(UnsupportedError);
  if (PyModule_AddObject(m, "UnsupportedError", UnsupportedError) < 0) {
    Py_DECREF(UnsupportedError);
    Py_DECREF(m);
    return NULL;
  }
  if (PyModule_AddIntConstant(m, "FAST", WLL_LEVEL_FAST) < 0 ||
      PyModule_AddIntConstant(m, "DEFAULT", WLL_LEVEL_DEFAULT) < 0 ||
      PyModule_AddIntConstant(m, "MAX_DIMENSION", WLL_MAX_DIMENSION) < 0) {
    Py_DECREF(m);
    return NULL;
  }
  return m;
}
