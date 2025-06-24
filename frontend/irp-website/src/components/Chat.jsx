import { useState } from "react";
import axios from "axios";

export default function Chat() {
  const [input, setInput] = useState("");
  // const [response, setResponse] = useState(null);
  // const [userQuestion, setUserQuestion] = useState("");
  const [messages, setMessages] = useState([]);

  const handleSubmit = async () => {
    try {
      const userMessage = { type: "user", text: input };
      setMessages((prev) => [...prev, userMessage]);
      // setUserQuestion(input);
      setInput("");

      const res = await axios.post("http://127.0.0.1:5000/apis/query", {
        user_query: input,
      });

      const botMessage = {
        type: "bot",
        text: typeof res.data.data === "string" ? res.data.data : "No response",
      };
      setMessages((prev) => [...prev, botMessage]);

      // setResponse(res.data);
    } catch (error) {
      console.error("Error:", error);
      const errorMessage = { type: "bot", text: error.message };
      setMessages((prev) => [...prev, errorMessage]);

      // setResponse({ status: error, message: error.message });
    }
  };

  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      console.log("Uploaded file:", file);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="w-full max-w-3xl px-4">
        {messages.length === 0 && (
          <main className="p-6">
            <h1 className="text-2xl font-semibold">Welcome to IC-RAG</h1>
          </main>
        )}

        {/* Chat messages */}
        <div className="mt-4 overflow-y-auto max-h-180">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.type === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`p-4 border rounded max-w-sm ${
                  message.type === "user" ? "bg-gray-50" : "bg-blue-50"
                }`}
              >
                <p className="text-sm font-medium text-left leading-loose">
                  {message.text}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Input Container */}
        <div className="border rounded-xl shadow-sm overflow-hidden bg-white mt-4">
          {/* chat box */}
          <input
            type="text"
            placeholder="Ask anything..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key == "Enter") {
                handleSubmit();
              }
            }}
            className="w-full px-4 py-3 outline-none"
          />

          {/* tool box */}
          <div className="flex items-center px-4 py-2">
            <button className="text-gray-500 mr-2 hover:bg-gray-200 rounded py-2 px-2">
              <label htmlFor="file-upload" className="cursor-pointer">
                +
              </label>
              <input
                id="file-upload"
                type="file"
                accept=".pdf,.ipynb,.py,.js,.jsx,.java,.cpp,.c,.h,.png,.jpg,.jpeg,.txt,.css,.json"
                onChange={handleFileUpload}
                className="hidden"
              />
            </button>
            <button className="text-gray-500 mr-2 hover:bg-gray-200 rounded py-2 px-2">
              Tools
            </button>

            <div className="flex-grow" />

            <button
              className="text-gray-500 hover:bg-gray-200 rounded py-2 px-2"
              onClick={handleSubmit}
            >
              Submit
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
