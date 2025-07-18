import axios from "axios";
import { useEffect, useState, useRef } from "react";

export default function Navbar({ onToggleSidebar }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [filesOpen, setFilesOpen] = useState(false);
  const [files, setFiles] = useState([]);
  const [deleting, setDeleting] = useState(null);
  const filesRef = useRef(null);

  useEffect(() => {
    function handleClickOutside(event) {
      if (filesRef.current && !filesRef.current.contains(event.target)) {
        setFilesOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const fetchFiles = async () => {
    try {
      console.log("Fetching files...");
      const response = await axios.get(
        "http://127.0.0.1:5000/apis/uploaded-files"
      );

      const filesData = response.data.files || [];

      setFiles(filesData);
    } catch (err) {
      console.error("Error fetching files:", err);
      setFiles([]);
    }
  };

  const deleteFile = async (__filename, collection) => {
    if (
      !confirm(
        `Are you sure you want to delete "${__filename}"? This will permanently remove all the relevant data from both vector and graph database.`
      )
    ) {
      return;
    }

    try {
      setDeleting(__filename);

      const response = await axios.delete(
        `http://127.0.0.1:5000/apis/delete-uploaded-files/${encodeURIComponent(
          __filename
        )}`,
        {
          data: { collection: collection },
        }
      );

      if (response.status == 200) {
        setFiles(files.filter((file) => file.name !== __filename));
        alert("File deleted.");
      }
    } catch (err) {
      console.error("Error deleting file:", err);
      alert(
        "Failed to delete file: " + (err.response?.data?.error || err.message)
      );
    } finally {
      setDeleting(null);
    }
  };

  const handleFilesClick = () => {
    setFilesOpen(!filesOpen);
    if (!filesOpen) fetchFiles();
  };

  return (
    <nav className="bg-white border px-6 py-4 w-full">
      <div className="flex items-center justify-between w-full">
        <div className="flex items-center space-x-4">
          <button
            onClick={onToggleSidebar}
            className="text-gray-800 text-2xl focus:outline-none hover:bg-gray-200 rounded px-4 py-2"
            aria-label="Toggle sidebar"
          >
            ☰
          </button>

          <div className="text-xl font-bold text-gray-800">IC-RAG</div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="relative" ref={filesRef}>
            <div
              className="bold text-2xl cursor-pointer hover:bg-gray-200 rounded px-4 py-2"
              onClick={handleFilesClick}
            >
              Files
            </div>

            {filesOpen && (
              <div className="absolute top-full right-0 mt-2 w-80 bg-white border rounded-lg shadow-lg z-50 max-h-60 overflow-y-auto">
                <div className="p-3 border-b">
                  <h3 className="font-semibold">Files</h3>
                </div>
                <div className="p-2">
                  {files.length === 0 ? (
                    <div className="text-gray-500 text-sm p-2">No files</div>
                  ) : (
                    files.map((file, index) => (
                      <div
                        key={file.name || index}
                        className="p-2 hover:bg-gray-50 rounded text-sm border-b last:border-b-0"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="font-medium truncate">
                              {file.name}
                            </div>
                            <div className="text-xs text-gray-500 mt-1">
                              Collection: {file.collection}, Type: {file.type},
                              Chunks: {file.chunks_count}
                            </div>
                            {file.size > 0 && (
                              <div className="text-xs text-gray-400">
                                Size: {(file.size / 1024).toFixed(2)} KB
                              </div>
                            )}
                          </div>
                          <button
                            onClick={() =>
                              deleteFile(file.name, file.collection)
                            }
                            disabled={deleting === file.name}
                            className="ml-2 px-2 py-1 text-xs bg-red-500 text-white rounded hover:bg-red disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
                            title="Delete file"
                          >
                            {deleting === file.name ? "Deleting..." : "Delete"}
                          </button>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Desktop login button */}
          <div className="hidden md:block">
            <button className="px-4 py-2 text-white bg-blue-600 rounded hover:bg-blue-700">
              Login
            </button>
          </div>

          {/* Mobile hamburger button */}
          <div className="md:hidden">
            <button
              onClick={() => setMenuOpen(!menuOpen)}
              className="text-gray-800 focus:outline-none"
              aria-label="Toggle login menu"
            >
              ☰
            </button>
          </div>
        </div>
      </div>

      {/* Mobile dropdown menu */}
      {menuOpen && (
        <div className="mt-2 md:hidden">
          <button className="w-full px-4 py-2 text-left text-white bg-blue-600 rounded hover:bg-blue-700">
            Login
          </button>
        </div>
      )}
    </nav>
  );
}
