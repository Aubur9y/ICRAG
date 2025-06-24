import { useState } from "react";

export default function Sidebar() {
  const [chats] = useState(["Chat 1", "Chat 2", "Chat 3", "Chat 4"]);

  const [selectedTitle, setSelectedTitle] = useState(null);

  return (
    <div className="fixed top-0 left-0 h-screen w-64 bg-white p-4 flex flex-col border-r border-gray-200 z-10">
      <button className="mb-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
        + New Chat
      </button>

      <div className="flex-1 overflow-y-auto space-y-2">
        {chats.map((chat) => (
          <div
            key={chat}
            onClick={() => setSelectedTitle(chat)}
            className={`p-3 rounded cursor-pointer hover:bg-gray-100 ${
              chat === selectedTitle ? "bg-gray-200" : ""
            }`}
          >
            {chat}
          </div>
        ))}
      </div>
    </div>
  );
}
