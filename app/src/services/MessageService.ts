import Answer from '../models/Answer';
import Message from '../models/Message';
const fromAnswer = (answer: Answer, prompt: string): Message => {
    return {
        id: Date.now(),
        prompt,
        completion: answer.response,
    }
}
const MessageService = {
    fromAnswer
};
export default MessageService;
