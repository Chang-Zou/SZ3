import struct

# Input and output file names
input_file = "../vx.f32"
output_file = "../vx_modifedFixSize.f32"

# Number of points to extract
num_points = 280953864
data_type = "f"  # float32

try:
    # Read the binary data
    with open(input_file, "rb") as f:
        # Read the first 25 float32 values
        data = f.read(num_points * struct.calcsize(data_type))

    # Write these values to the new binary file
    with open(output_file, "wb") as f:
        f.write(data)

    print(f"Successfully created {output_file} with the first {num_points} points.")

except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
