#!/usr/bin/env python3
# Copyright 2026 Anlatan Inc.
# SPDX-License-Identifier: Apache-2.0

"""Size comparison over a directory of images, in parallel.

For every API WebP in --src: decode it to PAM, encode the PAM with
`cwebp -lossless -exact -z 2` and with wllenc, check that wllenc's output
decodes to the identical RGBA, and record the VP8L payload sizes. Speed is
not measured here (jobs run concurrently); use roundtrip/bench_cwebp.py.
"""
import argparse
import glob
import os
import struct
import subprocess
import sys
import tempfile
from multiprocessing import Pool


def vp8l_size(data):
    pos = 12
    while pos + 8 <= len(data):
        (size,) = struct.unpack("<I", data[pos + 4:pos + 8])
        if data[pos:pos + 4] == b"VP8L":
            return size
        pos += 8 + size + (size & 1)
    raise ValueError("no VP8L chunk")


def run(job):
    path, libwebp, wllenc, wll_args, tmpdir = job
    name = os.path.basename(path)[:-5]
    with tempfile.TemporaryDirectory(dir=tmpdir) as d:
        pam = os.path.join(d, "in.pam")
        cw = os.path.join(d, "c.webp")
        ww = os.path.join(d, "w.webp")
        back = os.path.join(d, "back.pam")
        dwebp = os.path.join(libwebp, "dwebp")
        subprocess.run([dwebp, "-quiet", "-pam", path, "-o", pam], check=True)
        subprocess.run([os.path.join(libwebp, "cwebp"), "-quiet", "-lossless",
                        "-exact", "-z", "2", "-metadata", "none", pam, "-o", cw],
                       check=True)
        subprocess.run([wllenc] + wll_args + [pam, ww], check=True,
                       capture_output=True)
        subprocess.run([dwebp, "-quiet", "-pam", ww, "-o", back], check=True)
        exact = open(back, "rb").read() == open(pam, "rb").read()
        hdr = open(pam, "rb").read(200).decode("latin-1")
        w = int(hdr.split("WIDTH ")[1].split()[0])
        h = int(hdr.split("HEIGHT ")[1].split()[0])
        return (name, w * h, vp8l_size(open(path, "rb").read()),
                vp8l_size(open(cw, "rb").read()),
                vp8l_size(open(ww, "rb").read()), exact)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--libwebp", required=True)
    ap.add_argument("--wllenc", required=True)
    ap.add_argument("--wll-args", default="",
                    help="extra wllenc arguments, e.g. -fast")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="bench_full.csv")
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.src, "*.webp")))
    jobs = [(f, args.libwebp, args.wllenc, args.wll_args.split(), args.tmp)
            for f in files]
    rows = []
    with Pool(args.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(run, jobs, chunksize=4)):
            rows.append(r)
            if (i + 1) % 200 == 0:
                print("%d/%d" % (i + 1, len(jobs)), file=sys.stderr)
    rows.sort()
    with open(args.out, "w") as f:
        f.write("id,pixels,api,cwebp_z2,wll,exact\n")
        for r in rows:
            f.write("%s,%d,%d,%d,%d,%d\n" % r)
    px = sum(r[1] for r in rows)
    api = sum(r[2] for r in rows)
    cw = sum(r[3] for r in rows)
    wl = sum(r[4] for r in rows)
    bad = [r[0] for r in rows if not r[5]]
    ratios = sorted(r[4] / r[3] for r in rows)
    print("%d images, %.1f MP; wll output exact for %d/%d" %
          (len(rows), px / 1e6, len(rows) - len(bad), len(rows)))
    print("| encoder | bpp | size vs API |")
    print("|---|---|---|")
    print("| API | %.3f | 100.0%% |" % (api * 8 / px))
    print("| cwebp -z 2 | %.3f | %.2f%% |" % (cw * 8 / px, 100. * cw / api))
    print("| wll | %.3f | %.2f%% |" % (wl * 8 / px, 100. * wl / api))
    print("wll/cwebp per image: median %+.2f%%, p10 %+.2f%%, p90 %+.2f%%, "
          "worst %+.2f%%" % tuple(100 * (x - 1) for x in (
              ratios[len(ratios) // 2], ratios[len(ratios) // 10],
              ratios[len(ratios) * 9 // 10], ratios[-1])))
    if bad:
        print("NOT EXACT:", " ".join(bad))


if __name__ == "__main__":
    main()
