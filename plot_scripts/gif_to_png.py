import os
from PIL import Image
from rd3g.utilities.util import BASEDIR


def extract_frames(gif_path, output_folder):
    """
    Extracts individual frames from a GIF and saves them as PNGs.
    """
    # 1. Open the GIF
    try:
        with Image.open(gif_path) as im:
            # Create output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                print(f"Created directory: {output_folder}")

            frame_number = 0

            # Loop through frames
            while True:
                # 2. Construct the output filename (e.g., frame_001.png)
                # zfill(3) ensures 001, 002, etc. for sorting
                filename = f"frame_{str(frame_number).zfill(3)}.png"
                filepath = os.path.join(output_folder, filename)

                # 3. Save the frame
                # .convert('RGBA') ensures transparency is preserved if present
                im.convert('RGBA').save(filepath)
                print(f"Saved {filename}")

                frame_number += 1

                # 4. Move to next frame
                try:
                    im.seek(frame_number)
                except EOFError:
                    # End of GIF reached
                    break

            print(f"\nDone! Extracted {frame_number} frames to '{output_folder}'")

    except FileNotFoundError:
        print(f"Error: The file '{gif_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# --- Usage ---
if __name__ == "__main__":
    # Change these paths to match your files
    input_gif = os.path.join('gifs', 'intersection_8car.gif')
    output_dir = "extracted_frames"

    extract_frames(input_gif, output_dir)
