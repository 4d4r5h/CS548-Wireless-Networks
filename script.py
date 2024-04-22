import os
import subprocess

# Define the path to the FS folder
fs_folder = "FS"

# Get a list of all files in the FS folder
fs_files = [
    f for f in os.listdir(fs_folder) if os.path.isfile(os.path.join(fs_folder, f))
]

# Iterate over each file in the FS folder
for fs_file in fs_files:
    # Skip functionHO.py
    if fs_file == "functionHO.py":
        continue

    # Define the import statement for the current file
    import_statement = f"from FS.{os.path.splitext(fs_file)[0]} import jfs"

    # Define the optimization name for the current file
    optimization_name = os.path.splitext(fs_file)[0].upper()

    # Read the template.py file
    with open("template.py", "r") as template_file:
        template_content = template_file.readlines()

    # Make changes to the template content
    template_content[7] = import_statement + "  # Change here\n"
    template_content[11] = f"optimization = '{optimization_name}'  # Change here\n"

    # Write the modified content to a new file
    with open("modified_template.py", "w") as modified_template_file:
        modified_template_file.writelines(template_content)

    # Run the modified template.py script
    subprocess.run(["python", "modified_template.py"])

# Remove the modified_template.py file
os.remove("modified_template.py")
