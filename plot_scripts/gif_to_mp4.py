import argparse
import subprocess
from pathlib import Path


def gif_to_mp4(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(output_path),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a GIF to an MP4.")
    parser.add_argument("input_gif", type=Path)
    parser.add_argument("output_mp4", nargs="?", type=Path)
    args = parser.parse_args()

    output_path = args.output_mp4 or args.input_gif.with_suffix(".mp4")
    gif_to_mp4(args.input_gif, output_path)


if __name__ == "__main__":
    main()
