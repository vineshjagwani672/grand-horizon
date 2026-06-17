function MessageBubble({ message }) {
  const isUser = message.sender === "user";

  return (
    <div className={`message-row ${isUser ? "user-row" : "bot-row"}`}>
      <div className={`message-bubble ${isUser ? "user-bubble" : "bot-bubble"}`}>
        {message.text}
      </div>
    </div>
  );
}

export default MessageBubble;
