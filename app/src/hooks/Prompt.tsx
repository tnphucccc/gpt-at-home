import React, { createContext, ReactNode, useContext, useEffect, useState } from "react";
import Prompt from "../models/Prompt";
import PromptService from "../services/PromptService";
import Message from "../models/Message";
import MessageService from "../services/MessageService";

interface PromptContextType {
    prompt: Prompt | null;
    setPrompt: React.Dispatch<React.SetStateAction<Prompt | null>>;
    messages: Message[];
    loading: boolean;
    addMsg: (msg: Message) => void;
}



const PromptContext = createContext<PromptContextType | undefined>(undefined);

interface PromptProviderProps {
    children: ReactNode;
}

export const PromptProvider: React.FC<PromptProviderProps> = ({ children }) => {
    const [prompt, setPrompt] = useState<Prompt | null>(null);
    const [loading, setLoading] = useState(false)
    const [messages, setMessages] = useState<Message[]>([])
    useEffect(() => {
        const handler = async () => {
            if (prompt) {
                setLoading(true)
                const [response, error] = await PromptService.post(prompt)
                setLoading(false)
                if (error) {
                    alert(error.message)
                    return;
                }

                const newMsg = MessageService.fromAnswer(response, prompt.prompt)
                addMsg(newMsg)
            }
        }
        handler()
    }, [prompt])

    const addMsg = (msg: Message) => {
        setMessages((prevMsgs) => [...prevMsgs, msg])
    }

    return (
        <PromptContext.Provider value={{ prompt, setPrompt, messages, addMsg, loading }}>
            {children}
        </PromptContext.Provider>
    );
};

// Custom hook to use the context
export const usePrompt = () => {
    const context = useContext(PromptContext);
    if (!context) {
        throw new Error('usePrompt must be used within a PromptProvider');
    }
    return context;
};