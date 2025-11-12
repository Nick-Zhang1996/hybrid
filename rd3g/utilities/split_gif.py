import os
import sys
from PIL import Image

def gif_to_png_sequence(gif_path):
    # Ensure the file exists
    if not os.path.isfile(gif_path):
        print("Error: File not found.")
        return
    
    # Extract the directory and filename without extension
    dir_name = os.path.dirname(gif_path)
    base_name = os.path.splitext(os.path.basename(gif_path))[0]
    output_folder = os.path.join(dir_name, base_name)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the GIF file
    with Image.open(gif_path) as gif:
        frame_number = 0
        while True:
            frame_path = os.path.join(output_folder, f"frame_{frame_number:03d}.png")
            gif.save(frame_path, format="PNG")
            
            frame_number += 1
            
            try:
                gif.seek(frame_number)  # Move to the next frame
            except EOFError:
                break  # End of frames
    
    print(f"GIF converted successfully! Frames saved in '{output_folder}' folder.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py path/to/file.gif")
    else:
        gif_to_png_sequence(sys.argv[1])

