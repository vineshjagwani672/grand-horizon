import { useState } from "react";
import MessageBubble from "../components/MessageBubble.jsx";
import LoadingDots from "../components/LoadingDots.jsx";
import { sendMessage } from "../services/chatService.js";

function ChatPage() {
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Hello! Ask me anything from the Grand Horizon Hotel knowledge base."
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(event) {
    event.preventDefault();

    const question = input.trim();
    if (!question || loading) return;

    setMessages((current) => [...current, { sender: "user", text: question }]);
    setInput("");
    setLoading(true);

    try {
      const data = await sendMessage(question);
      setMessages((current) => [...current, { sender: "bot", text: data.answer }]);
    } catch (error) {
      const message =
        error.response?.data?.detail || "Sorry, something went wrong. Please try again.";
      setMessages((current) => [...current, { sender: "bot", text: message }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="chat-panel">
        <header className="chat-header">
          <div>
            <p className="eyebrow">RAG Course Project</p>
            <h1>Grand Horizon Hotel Chatbot</h1>
          </div>
          <span className="status-pill">PDF Ready</span>
        </header>

        <div className="messages">
          {messages.map((message, index) => (
            <MessageBubble key={`${message.sender}-${index}`} message={message} />
          ))}
          {loading && <LoadingDots />}
        </div>

        <form className="chat-form" onSubmit={handleSubmit}>
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask about menu, hours, reservations..."
            aria-label="Ask a question"
          />
          <button type="submit" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </section>
    </main>
  );
}

export default ChatPage;
