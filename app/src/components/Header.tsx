import React from 'react';
import { usePrompt } from '../hooks/Prompt';

const Header: React.FC = () => {
    const { loading } = usePrompt();

    return (
        <div className="w-full text-center text-white px-4">
            <p className="italic text-gray-300 py-1">
                A tiny character-level GPT trained only on Shakespeare. It continues the scene you
                start — it can't answer questions or follow instructions.
            </p>
            {loading && (
                <p role="status" className="text-white py-1">Writing the scene...</p>
            )}
        </div>
    );
};

export default Header;
