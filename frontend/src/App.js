import React, { useState, useRef, useEffect } from 'react';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setIsLoading(true);

    // Get analysis for the user message (using mock functions)
    const analysis = {
      expanded_query: getMockExpandedQuery(userMessage, messages),
      topic: getMockTopic(userMessage)
    };

    // Add user message with analysis
    const newUserMessage = {
      id: Date.now(),
      role: 'user',
      content: userMessage,
      analysis: analysis
    };

    setMessages(prev => [...prev, newUserMessage]);

    try {
      // Call Gemini API for assistant response
      const assistantResponse = await getGeminiResponse([...messages, newUserMessage]);
      
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: assistantResponse,
        analysis: null
      }]);
    } catch (error) {
      console.error('Error calling Gemini API:', error);
      // Fallback to mock response if Gemini API fails
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: "I apologize, but I'm having trouble connecting right now. Please check your API key configuration.",
        analysis: null
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Call Gemini API for assistant responses
  const getGeminiResponse = async (conversationHistory) => {
    const apiKey = process.env.REACT_APP_GEMINI_API_KEY;
    
    if (!apiKey) {
      throw new Error('Gemini API key not configured. Please set REACT_APP_GEMINI_API_KEY in your .env file');
    }

    // Format conversation history for Gemini API
    const contents = conversationHistory.map(msg => ({
      role: msg.role === 'user' ? 'user' : 'model',
      parts: [{ text: msg.content }]
    }));

    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey}`;
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: contents
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error?.message || `API error: ${response.status}`);
    }

    const data = await response.json();
    
    if (data.candidates && data.candidates[0] && data.candidates[0].content) {
      return data.candidates[0].content.parts[0].text;
    }
    
    throw new Error('Invalid response format from Gemini API');
  };

  // Mock functions for demo without backend
  const getMockExpandedQuery = (query, history) => {
    const lastAssistant = history.filter(m => m.role === 'assistant').pop();
    if (query.toLowerCase().includes('that') || query.toLowerCase().includes('it')) {
      return `What about ${lastAssistant?.content?.split(' ').slice(0, 5).join(' ') || 'the topic'}...?`;
    }
    if (query.toLowerCase().includes('him') || query.toLowerCase().includes('her')) {
      return query.replace(/him|her|them/gi, '[referenced person]');
    }
    return query;
  };

  const getMockTopic = (query) => {
    const q = query.toLowerCase();
    if (q.includes('movie') || q.includes('film') || q.includes('actor')) {
      return { level_1: 'Entertainment', level_2: 'Movies' };
    }
    if (q.includes('football') || q.includes('soccer') || q.includes('goal')) {
      return { level_1: 'Sports', level_2: 'Football' };
    }
    if (q.includes('code') || q.includes('programming') || q.includes('software')) {
      return { level_1: 'Technology', level_2: 'Software Development' };
    }
    if (q.includes('health') || q.includes('doctor') || q.includes('medicine')) {
      return { level_1: 'Health', level_2: 'Medicine' };
    }
    return { level_1: 'General', level_2: 'Chitchat' };
  };


  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Query Expansion & Topic Tagging</h1>
        <p>Chat interface with real-time query analysis</p>
        <button className="clear-btn" onClick={clearChat}>Clear Chat</button>
      </header>

      <main className="chat-container">
        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome-message">
              <h2>Welcome!</h2>
              <p>Start a conversation to see query expansion and topic classification in action.</p>
              <div className="example-queries">
                <p><strong>Try asking:</strong></p>
                <ul>
                  <li>"Tell me about the latest Marvel movie"</li>
                  <li>"Who scored the winning goal?" (after discussing football)</li>
                  <li>"What about him?" (after mentioning a person)</li>
                </ul>
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`message-row ${message.role}`}>
              {message.role === 'user' && message.analysis && (
                <div className="analysis-card">
                  <div className="analysis-item">
                    <div className="analysis-label">Expanded Query</div>
                    <div className="analysis-value">{message.analysis.expanded_query}</div>
                  </div>
                  <div className="analysis-item">
                    <div className="analysis-label">Topic</div>
                    <div className="topic-tags">
                      <span className="topic-tag level1">{message.analysis.topic.level_1}</span>
                      <span className="topic-separator">→</span>
                      <span className="topic-tag level2">{message.analysis.topic.level_2}</span>
                    </div>
                  </div>
                </div>
              )}
              <div className="avatar">
                {message.role === 'user' ? (
                  <div className="avatar-user">🐿️</div>
                ) : (
                  <div className="avatar-assistant">🤖</div>
                )}
              </div>
              <div className="message-content-wrapper">
                <div className={`message-bubble ${message.role}`}>
                  {message.content}
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="message-row assistant">
              <div className="avatar">
                <div className="avatar-assistant">🤖</div>
              </div>
              <div className="message-content-wrapper">
                <div className="message-bubble assistant loading">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form className="input-form" onSubmit={handleSubmit}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            Send
          </button>
        </form>
      </main>

      <footer className="footer">
        <p>Query Expansion & Topic Tagging Demo</p>
      </footer>
    </div>
  );
}

export default App;
