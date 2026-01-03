import React, { useState, useRef, useEffect } from 'react';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'assistant',
      content: "Hello! How can I help you with your UPSC preparation today? Do you have any questions about the exam, strategy, or any specific topics you're studying?"
    }
  ]);
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

    // Get analysis for the user message
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
      // Call backend API for analysis
      const response = await fetchAnalysis([...messages, newUserMessage]);

      // Add assistant response if provided
      if (response.assistant_response) {
        setMessages(prev => [...prev, {
          id: Date.now() + 1,
          role: 'assistant',
          content: response.assistant_response
        }]);
      }
    } catch (error) {
      console.error('Error:', error);
      // Add mock assistant response
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'assistant',
        content: getMockAssistantResponse(userMessage)
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  // Mock function - replace with actual API call
  const fetchAnalysis = async (conversationHistory) => {
    const response = await fetch('http://localhost:8000/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: conversationHistory })
    });

    if (!response.ok) throw new Error('API error');
    return response.json();
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

  const getMockAssistantResponse = (query) => {
    return "I understand your question. Let me help you with that. This is a demo response - connect the backend for real responses.";
  };

  const clearChat = () => {
    setMessages([
      {
        id: 1,
        role: 'assistant',
        content: "Hello! How can I help you with your UPSC preparation today? Do you have any questions about the exam, strategy, or any specific topics you're studying?"
      }
    ]);
  };

  return (
    <div className="app">
      {/* Header */}
      <div className="header">
        <h1>Query Expansion and Topic Tagging</h1>
        <button className="clear-chat-btn" onClick={clearChat}>
          Clear Chat
        </button>
      </div>

      {/* Chat Messages */}
      <div className="chat-container">
        <div className="messages">
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

        {/* Input Bar */}
        <form className="input-bar" onSubmit={handleSubmit}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask me anything"
            disabled={isLoading}
            className="chat-input"
          />
          <button 
            type="submit" 
            className="attach-btn"
            disabled={isLoading || !input.trim()}
          >
            +
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
