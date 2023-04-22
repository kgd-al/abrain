from pathlib import Path


def example_path(name):
    output_folder = Path("tmp/examples/")
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    return output_folder.joinpath(name)
