export default interface Message {
    id: number;
    // Text the user started the scene with (may be empty)
    prompt: string;
    // Text the model generated to continue the prompt
    completion: string;
}
