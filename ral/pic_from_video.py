#!/usr/bin/env python3
"""
Video Frame Extraction Script

Extracts a single frame from a video file and saves it as an image.
Supports various video formats and allows specification of frame number or time.
"""

import argparse
import cv2
import os
import sys
from pathlib import Path


def extract_frame(video_path, output_path, frame_number=None, time_seconds=None):
    """
    Extract a frame from a video and save it as an image.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str): Path where the extracted frame will be saved
        frame_number (int, optional): Frame number to extract (0-indexed)
        time_seconds (float, optional): Time in seconds to extract frame from
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return False
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Video info: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s duration")
    
    # Determine which frame to extract
    target_frame = 0
    
    if time_seconds is not None:
        # Convert time to frame number
        target_frame = int(time_seconds * fps)
        print(f"Extracting frame at {time_seconds}s (frame {target_frame})")
    elif frame_number is not None:
        target_frame = frame_number
        print(f"Extracting frame {frame_number}")
    else:
        # Default to middle frame
        target_frame = frame_count // 2
        print(f"No frame/time specified, extracting middle frame ({target_frame})")
    
    # Validate frame number
    if target_frame < 0 or target_frame >= frame_count:
        print(f"Error: Frame {target_frame} is out of range (0-{frame_count-1})")
        cap.release()
        return False
    
    # Set video position to target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    
    # Read the frame
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Could not read frame {target_frame}")
        cap.release()
        return False
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the frame
    success = cv2.imwrite(output_path, frame)
    
    if success:
        print(f"Frame successfully saved to '{output_path}'")
        print(f"Image dimensions: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print(f"Error: Could not save frame to '{output_path}'")
    
    # Clean up
    cap.release()
    
    return success


def main():
    """Main function to handle command line arguments and execute frame extraction."""
    parser = argparse.ArgumentParser(
        description="Extract a frame from a video file and save it as an image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract middle frame from video
  python pic_from_video.py input.mp4 output.jpg
  
  # Extract specific frame number
  python pic_from_video.py input.mp4 output.png --frame 100
  
  # Extract frame at specific time
  python pic_from_video.py input.mp4 output.jpg --time 30.5
  
  # Extract frame from video in videos directory
  python pic_from_video.py videos/experiment.mp4 images/frame.jpg --time 10
        """
    )
    
    parser.add_argument(
        'video_path',
        help='Path to the input video file'
    )
    
    parser.add_argument(
        'output_path',
        nargs='?',
        help='Path where the extracted frame will be saved (e.g., frame.jpg, frame.png)'
    )
    
    parser.add_argument(
        '-f', '--frame',
        type=int,
        help='Frame number to extract (0-indexed). Cannot be used with --time.'
    )
    
    parser.add_argument(
        '-t', '--time',
        type=float,
        help='Time in seconds to extract frame from. Cannot be used with --frame.'
    )
    
    parser.add_argument(
        '--info-only',
        action='store_true',
        help='Only display video information without extracting a frame'
    )
    
    args = parser.parse_args()
    
    # Check for mutually exclusive arguments
    if args.frame is not None and args.time is not None:
        print("Error: Cannot specify both --frame and --time arguments.")
        sys.exit(1)
    
    # Check if output_path is required but missing
    if not args.info_only and args.output_path is None:
        print("Error: output_path is required when not using --info-only.")
        sys.exit(1)
    
    # If info-only mode, just display video info
    if args.info_only:
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{args.video_path}'.")
            sys.exit(1)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"Video: {args.video_path}")
        print(f"Dimensions: {width}x{height}")
        print(f"Frames: {frame_count}")
        print(f"FPS: {fps:.2f}")
        print(f"Duration: {duration:.2f} seconds")
        
        cap.release()
        return
    
    # Extract the frame
    success = extract_frame(
        args.video_path,
        args.output_path,
        frame_number=args.frame,
        time_seconds=args.time
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()