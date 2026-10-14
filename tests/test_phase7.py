from app import ensure_dirs, list_supported_files


def test_app_can_import_and_list_supported_files_without_indexed_data():
    ensure_dirs()
    files = list_supported_files()
    assert isinstance(files, list)
    assert "data" not in files
