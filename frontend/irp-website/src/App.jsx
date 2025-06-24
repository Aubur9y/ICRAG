import { useState } from "react";
import "./App.css";
import Navbar from "./components/NavBar.jsx";
import Sidebar from "./components/Sidebar.jsx";
import Chat from "./components/Chat.jsx";

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="h-screen overflow-hidden">
      {sidebarOpen && <Sidebar />}

      <div
        className={`flex flex-col h-full transition-all duration-300 ${
          sidebarOpen ? "ml-64" : "ml-0"
        }`}
      >
        <Navbar onToggleSidebar={() => setSidebarOpen(!sidebarOpen)} />
        
        <div className="flex-1">
          <Chat />
        </div>
      </div>
    </div>
  );
}

export default App;
