import React, { useEffect, useRef } from "react";
import MessageComponent from "./Message";
import { usePrompt } from "../hooks/Prompt";
import SendMessage from "./SendMessage";


const ChatBox: React.FC = () => {
  const { messages, loading, prompt } = usePrompt();
  const messagesBottomDiv = useRef<HTMLDivElement | null>(null);
  const scrollToBottom = () => {
    if (!messagesBottomDiv || !messagesBottomDiv.current) return;
    messagesBottomDiv.current.scrollTop =
      messagesBottomDiv.current?.scrollHeight;
  };
  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);
  return (
    <div className="flex flex-col h-[90vh]">
      <div
        className="w-full grow overflow-y-auto space-y-4 py-2"
        ref={messagesBottomDiv}
        role="log"
        aria-live="polite"
      >
        {messages.length === 0 && !loading && (
          <p className="text-center text-gray-400 pt-8 px-4">
            Pick a character below or type the first line of a scene, then press send.
          </p>
        )}
        {messages.map((message) => (
          <MessageComponent key={message.id} message={message} />
        ))}
        {loading && prompt && (
          <MessageComponent message={{ id: 0, prompt: prompt.prompt, completion: "" }} pending />
        )}
      </div>
      <div className="w-full py-2 flex-none">
        <SendMessage />
      </div>
    </div>
  );
};


export default ChatBox;
