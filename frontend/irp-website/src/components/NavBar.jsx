import { useState } from "react";

export default function Navbar({ onToggleSidebar }) {
  const [menuOpen, setMenuOpen] = useState(false);

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
