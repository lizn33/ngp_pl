#!/usr/bin/env python3

# From nvidia instant_ngp scripts/colmap2nerf.py

import argparse
from glob import glob
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil

def parse_args():
	parser = argparse.ArgumentParser(description="Convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place.")

	parser.add_argument("--video_in", default="", help="Run ffmpeg first to convert a provided video file into a set of images. Uses the video_fps parameter also.")
	parser.add_argument("--video_fps", default=2)
	parser.add_argument("--time_slice", default="", help="Time (in seconds) in the format t1,t2 within which the images should be generated from the video. E.g.: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video.")
	parser.add_argument("--run_colmap", default=True, help="run colmap first on the image folder")
	parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="Select which matcher colmap should use. Sequential for videos, exhaustive for ad-hoc images.")
	parser.add_argument("--colmap_db", default="dataset/colmap.db", help="colmap database filename")
	parser.add_argument("--colmap_camera_model", default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"], help="Camera model")
	parser.add_argument("--colmap_camera_params", default="", help="Intrinsic parameters, depending on the chosen model. Format: fx,fy,cx,cy,dist")
	parser.add_argument("--images", default="images", help="Input path to the images.")
	parser.add_argument("--sparse", default="dataset/sparse", help="Input path to the colmap sparse files (set automatically if --run_colmap is used).")
	parser.add_argument("--text", default="dataset/colmap_text", help="Input path to the colmap text files (set automatically if --run_colmap is used).")
	parser.add_argument("--overwrite", action="store_true", help="Do not ask for confirmation for overwriting existing images and COLMAP data.")
	args = parser.parse_args()
	return args

def do_system(arg):
	print(f"==== running: {arg}")
	err = os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)

def run_ffmpeg(args):
	ffmpeg_binary = "ffmpeg"

	if not os.path.isabs(args.images):
		args.images = os.path.join(os.path.dirname(args.video_in), args.images)

	images = "\"" + args.images + "\""
	video =  "\"" + args.video_in + "\""
	fps = float(args.video_fps) or 1.0
	print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
	if not args.overwrite and (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
		sys.exit(1)
	try:
		# Passing Images' Path Without Double Quotes
		shutil.rmtree(args.images)
	except:
		pass
	do_system(f"mkdir {images}")

	time_slice_value = ""
	time_slice = args.time_slice
	if time_slice:
		start, end = time_slice.split(",")
		time_slice_value = f",select='between(t\,{start}\,{end})'"
	do_system(f"{ffmpeg_binary} -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}{time_slice_value}\" {images}/%04d.jpg")
	
def run_colmap(args):
	colmap_binary = "colmap"

	db = args.colmap_db
	images = "\"" + args.images + "\""
	db_noext=str(Path(db).with_suffix(""))

	if args.text=="text":
		args.text=db_noext+"_text"
	text=args.text
	sparse=args.sparse
	print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
	if not args.overwrite and (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
		sys.exit(1)
	if os.path.exists(db):
		os.remove(db)
	do_system(f"{colmap_binary} feature_extractor --ImageReader.camera_model {args.colmap_camera_model} --ImageReader.camera_params \"{args.colmap_camera_params}\" --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db} --image_path {images}")

	match_cmd = f"{colmap_binary} {args.colmap_matcher}_matcher --SiftMatching.guided_matching=true --database_path {db}"
	do_system(match_cmd)

	try:
		shutil.rmtree(sparse)
	except:
		pass
	do_system(f"mkdir {sparse}")
	do_system(f"{colmap_binary} mapper --database_path {db} --image_path {images} --output_path {sparse}")
	do_system(f"{colmap_binary} bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
	try:
		shutil.rmtree(text)
	except:
		pass
	do_system(f"mkdir {text}")
	do_system(f"{colmap_binary} model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")

if __name__ == "__main__":
	args = parse_args()
	if args.video_in != "":
		run_ffmpeg(args)
	if args.run_colmap:
		run_colmap(args)	