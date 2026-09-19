import Answer from "../models/Answer";
import Prompt from "../models/Prompt";
import axios, { AxiosRequestConfig } from 'axios';
import { Err, Ok, Result } from "../types";
const baseURL = import.meta.env.VITE_API_URL ?? "http://localhost:8000"
const config: AxiosRequestConfig = {
    headers: {
        'Content-Type': 'application/json',
        "accept": "application/json"
    }
}

const requestHandler = axios.create({
    baseURL: baseURL
});
const post = async (prompt: Prompt): Promise<Result<Answer>> => {
    const serializedPrompt = toJson(prompt);
    try {
        const response = await requestHandler.post<Answer>("/generate", serializedPrompt, config);
        return Ok(response.data);
    } catch (e) {
        // axios throws on network errors and non-2xx responses
        const detail = axios.isAxiosError(e) ? e.response?.data?.detail : undefined;
        if (typeof detail === "string") {
            return Err(new Error(detail));
        }
        return Err(new Error(`Could not reach the model server at ${baseURL}: ${(e as Error).message}`));
    }
}

const toJson = (prompt: Prompt) => {
    return JSON.stringify({
        prompt: prompt.prompt,
        max_tokens: prompt.maxTokens
    })
}

const PromptService = {
    post
};
export default PromptService;