def save_file(file, path):
    file_content = file.read()
    file_path = f"{path}/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path
