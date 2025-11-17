"""
Context manager for conversation history with windowing and summarization.
"""
import ollama
from rag import estimate_tokens


class ContextManager:
    """
    Manages conversation history with context window limits.
    Ensures resume context is always preserved while managing history size.
    """
    
    def __init__(self, max_history_tokens, min_recent_messages, summary_threshold, llm_model):
        self.max_history_tokens = max_history_tokens
        self.min_recent_messages = min_recent_messages
        self.summary_threshold = summary_threshold
        self.llm_model = llm_model
        self.history = []  # List of (role, message) tuples
        self.summary = None  # Summarized older conversation
        
    def add_exchange(self, user_message, bot_message):
        """
        Add a user-bot exchange to the history.
        """
        self.history.append(("User", user_message))
        self.history.append(("Bot", bot_message))
        self._manage_context()
    
    def _manage_context(self):
        """
        Manages context window by summarizing old messages when needed.
        Always preserves recent messages to maintain conversation flow.
        """
        # Calculate current history size
        history_text = self._format_history(self.history)
        history_tokens = estimate_tokens(history_text)
        
        # Check if we need to manage context
        if history_tokens <= self.max_history_tokens * self.summary_threshold:
            return  # No action needed
        
        # We need to summarize. Keep recent messages, summarize older ones
        recent_count = max(self.min_recent_messages, len(self.history) // 4)
        recent_messages = self.history[-recent_count:]
        old_messages = self.history[:-recent_count]
        
        if not old_messages:
            # Can't reduce further without losing recent context
            return
        
        # Create or update summary of old messages
        old_history_text = self._format_history(old_messages)
        if self.summary:
            # Combine existing summary with new old messages
            summary_prompt = (
                f"Previous conversation summary:\n{self.summary}\n\n"
                f"Additional conversation to add to summary:\n{old_history_text}\n\n"
                f"Create a concise summary that captures the key points from both the previous summary "
                f"and the additional conversation. Focus on facts, decisions, and important context. "
                f"Keep it brief (under 200 words)."
            )
        else:
            summary_prompt = (
                f"Summarize the following conversation, focusing on key facts, decisions, "
                f"and important context. Keep it brief (under 200 words):\n\n{old_history_text}"
            )
        
        try:
            response = ollama.generate(
                model=self.llm_model,
                prompt=summary_prompt,
                stream=False
            )
            self.summary = response['response'].strip()
            
            # Replace old messages with summary
            self.history = recent_messages
            
        except Exception as e:
            print(f"\nWarning: Could not summarize conversation history: {e}")
            # Fallback: just truncate to recent messages
            self.history = recent_messages
    
    def _format_history(self, messages):
        """
        Formats a list of (role, message) tuples into a history string.
        """
        return "\n".join([f"{role}: {message}" for role, message in messages])
    
    def get_formatted_history(self):
        """
        Returns the formatted conversation history, including summary if present.
        """
        parts = []
        if self.summary:
            parts.append(f"[Earlier conversation summary: {self.summary}]")
        if self.history:
            parts.append(self._format_history(self.history))
        return "\n".join(parts) if parts else ""
    
    def get_history_token_count(self):
        """
        Returns estimated token count of current history.
        """
        history_text = self.get_formatted_history()
        return estimate_tokens(history_text)

