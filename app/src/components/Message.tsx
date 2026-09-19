import React from "react";
import Message from "../models/Message";

const MessageComponent: React.FC<{ message: Message; pending?: boolean }> = ({ message, pending }) => {
  return (
    <div className="w-full max-w-3xl mx-auto px-4">
      <p className="whitespace-pre-wrap font-serif leading-relaxed text-white bg-[#2f2f2f] rounded-lg p-4">
        {message.prompt && <strong className="text-amber-300">{message.prompt}</strong>}
        {pending ? <span className="opacity-60 animate-pulse">…</span> : message.completion}
      </p>
    </div>
  );
};
export default MessageComponent;
