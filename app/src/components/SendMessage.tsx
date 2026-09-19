import React, { useState, useCallback } from "react";
import Slider from "@mui/material/Slider";
import SendIcon from "@mui/icons-material/Send";
import { usePrompt } from "../hooks/Prompt";
import Prompt from "../models/Prompt";

const SendMessage: React.FC = () => {
  const [message, setMessage] = useState<string>("");
  const [token, setToken] = useState<number>(500);
  const { setPrompt, loading } = usePrompt();

  const valuetext = useCallback((value: number) => {
    setToken(value);
    return `${value}`;
  }, []);

  const sendMessage = useCallback(
    async (event: React.FormEvent) => {
      event.preventDefault();
      if (loading) return;
      // An empty prompt is allowed: the model then starts a random scene
      const prompt: Prompt = {
        prompt: message,
        maxTokens: token,
      };
      setPrompt(prompt);
      setMessage("");
    },
    [message, token, loading, setPrompt]
  );

  return (
    <form
      className="w-full flex flex-col px-5 py-3 justify-center items-center"
      onSubmit={sendMessage}
    >
      <StarterPrompts onPick={setMessage} />
      <div className="w-full max-w-3xl py-3 rounded-lg px-5 bg-[#2f2f2f] flex">
        <div className="w-full">
          <MessageInput message={message} setMessage={setMessage} />
          <TokenSlider token={token} valuetext={valuetext} />
        </div>

        <SendButton loading={loading} />
      </div>
    </form>
  );
};

// Most frequent speakers in the Tiny Shakespeare training text
const STARTERS = [
  "ROMEO:\n",
  "JULIET:\n",
  "GLOUCESTER:\n",
  "KING RICHARD III:\nMy lord, ",
  "DUKE VINCENTIO:\n",
  "PETRUCHIO:\n",
];

const StarterPrompts: React.FC<{ onPick: (text: string) => void }> = React.memo(({ onPick }) => (
  <div className="w-full max-w-3xl flex flex-wrap gap-2 pb-2">
    {STARTERS.map((text) => (
      <button
        key={text}
        type="button"
        className="text-sm text-white border border-gray-500 rounded-full px-3 py-1 hover:bg-[#2f2f2f]"
        onClick={() => onPick(text)}
      >
        {text.split("\n")[0]}
      </button>
    ))}
  </div>
));

const MessageInput: React.FC<{
  message: string;
  setMessage: React.Dispatch<React.SetStateAction<string>>;
}> = React.memo(({ message, setMessage }) => (
  <textarea
    className="bg-transparent w-full focus:outline-none text-white py-2"
    aria-label="Start of the scene"
    rows={2}
    placeholder={"Start a scene, e.g. ROMEO:  — or leave empty for a random scene"}
    value={message}
    onChange={(e) => setMessage(e.target.value)}
  />
));

const TokenSlider: React.FC<{
  token: number;
  valuetext: (value: number) => string;
}> = React.memo(({ token, valuetext }) => (
  <div className="w-full flex">
    <p className="text-white w-1/3">Tokens: {token}</p>
    <div className="w-full flex justify-center items-center">
      <div className="w-4/5">
        <Slider
          aria-label="Token"
          defaultValue={token}
          getAriaValueText={valuetext}
          valueLabelDisplay="auto"
          shiftStep={100}
          step={100}
          marks
          min={100}
          max={2000}
        />
      </div>
    </div>
  </div>
));

const SendButton: React.FC<{ loading: boolean }> = React.memo(({ loading }) => (
  <button
    type="submit"
    className={`text-white p-2 ${
      loading ? "cursor-not-allowed opacity-50" : "cursor-pointer"
    }`}
    disabled={loading}
    aria-label="Generate"
  >
    <SendIcon />
  </button>
));

export default SendMessage;
