#!/usr/bin/env python3
"""Compare VP8L payload sizes and encode times of `cwebp -lossless -exact -z N`
against a directory of lossless WebP files.

Each --src/*.webp is decoded once to PAM (cached in --pam), then
re-encoded per level. Sizes are VP8L chunk payloads (no container/metadata),
times are cwebp's own "Time to encode picture" (single-threaded).
"""
import argparse
import glob
import os
import re
import struct
import subprocess
import sys

LIBWEBP = os.path.join(os.environ.get("LIBWEBP", "third_party/libwebp"), "build")


def vp8l_size(data):
    pos = 12
    while pos + 8 <= len(data):
        fourcc = data[pos:pos + 4]
        (size,) = struct.unpack("<I", data[pos + 4:pos + 8])
        if fourcc == b"VP8L":
            return size
        pos += 8 + size + (size & 1)
    raise ValueError("no VP8L chunk")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--pam", required=True)
    ap.add_argument("--levels", default="0,1,2,3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--libwebp", default=LIBWEBP)
    args = ap.parse_args()
    cwebp = os.path.join(args.libwebp, "cwebp")
    dwebp = os.path.join(args.libwebp, "dwebp")
    levels = [int(x) for x in args.levels.split(",")]
    os.makedirs(args.pam, exist_ok=True)
    tmp = os.path.join(args.pam, "bench.tmp.webp")

    files = sorted(glob.glob(os.path.join(args.src, "*.webp")))
    if args.limit:
        files = files[:args.limit]
    tot_px = 0
    tot_api = 0
    tot = {z: [0, 0.0] for z in levels}
    for f in files:
        data = open(f, "rb").read()
        api = vp8l_size(data)
        pam = os.path.join(args.pam, os.path.basename(f)[:-5] + ".pam")
        if not os.path.exists(pam):
            subprocess.run([dwebp, "-quiet", "-pam", f, "-o", pam + ".tmp"], check=True)
            os.replace(pam + ".tmp", pam)
        with open(pam, "rb") as p:
            hdr = p.read(200).decode("latin-1")
        w = int(re.search(r"WIDTH (\d+)", hdr).group(1))
        h = int(re.search(r"HEIGHT (\d+)", hdr).group(1))
        tot_px += w * h
        tot_api += api
        for z in levels:
            r = subprocess.run([cwebp, "-v", "-lossless", "-exact", "-z", str(z),
                                "-metadata", "none", pam, "-o", tmp],
                               check=True, capture_output=True, text=True)
            t = float(re.search(r"Time to encode picture: ([\d.]+)s", r.stderr).group(1))
            tot[z][0] += vp8l_size(open(tmp, "rb").read())
            tot[z][1] += t
    os.remove(tmp)

    mp = tot_px / 1e6
    print("%d images, %.1f MP" % (len(files), mp))
    print("| encoder | bpp | size vs API | MP/s |")
    print("|---|---|---|---|")
    print("| API | %.3f | 100.0%% | - |" % (tot_api * 8 / tot_px))
    for z in levels:
        size, t = tot[z]
        print("| cwebp -z %d | %.3f | %.1f%% | %.2f |"
              % (z, size * 8 / tot_px, 100.0 * size / tot_api, mp / t))


if __name__ == "__main__":
    main()
