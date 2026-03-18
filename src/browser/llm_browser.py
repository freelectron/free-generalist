from collections import deque
from time import time                                                                                                                                                       
                                                                                                                                                                            
from browser.llms.session import LLMSession                                                                                                                                 
from browser.llms.session import ChatGPT, DeepSeek, Gemini, Qwen, Claude, Mistral                                                                                           
from browser.search.web import BraveBrowser                                                                                                                                 
from browser.browser import ChromeBrowser                                                                                                                                   
from generalist.openclaw.tool_calling import add_tool_directive, parse_out_tool_call                                                                                        
from clog import get_logger
                                                                                                                                                                            
                                                                                                                                                                            
SMALL_CONTEXT_THRESHOLD = 32000                                                                                                                                             
INITIAL_TOKENS = 100                                                                                                                                                        
PENALTY_TOKENS = 10                                                                                                                                                         
TIMEOUT_HOURS = 4    


logger = get_logger(__name__)
                                                                                                                                                                            
                                                                                                                                                                            
class LLMBrowser:                                                                                                                                                           
    """                                                                                                                                                                     
    Manages a pool of LLM sessions. Each session has a token credit.
    On failure the credit is reduced by PENALTY_TOKENS. When credit drops                                                                                                   
    below zero the session is moved to a FIFO timeout queue for TIMEOUT_HOURS,                                                                                              
    then returned to the active pool.                                                                                                                                       
    """                                                                                                                                                                     
                                                                                                                                                                            
    def create_long_message_sessions(self) -> list[LLMSession]:                                                                                                             
        """Sessions that allow a high number of tokens in a single paste."""                                                                                                
        return [self.DEEPSEEK_SESSION, self.CLAUDE_SESSION, self.MISTRAL_SESSION, self.QWEN_SESSION]                                                                                            
                                                                                                                                                                            
    def create_small_message_sessions(self) -> list[LLMSession]:                                                                                                            
        """Sessions that can only (easily) handle a low number of tokens."""                                                                                                
        return [self.GEMINI_SESSION, self.CHATGPT_SESSION]                                                                                                                            
                                                                                                                                                                            
    def __init__(self):
        chrome_browser = ChromeBrowser()  
        # TODO: make it nicer 
        assert chrome_browser.chrome_user_data_dir

        self.CHATGPT_SESSION = ChatGPT(chrome_browser, session_id="closed_ai")                                                                                                           
        self.DEEPSEEK_SESSION = DeepSeek(chrome_browser, session_id="deep_seek")                                                                                                         
        self.GEMINI_SESSION = Gemini(chrome_browser, session_id="gemini")                                                                                                                
        self.QWEN_SESSION = Qwen(chrome_browser, session_id="qwen")                                                                                                                      
        self.CLAUDE_SESSION = Claude(chrome_browser, session_id="claude")                                                                                                                
        self.MISTRAL_SESSION = Mistral(chrome_browser, session_id="mistral")      

        # list of [session, tokens] — mutable so token credit updates in-place                                                                                              
        long_sessions = self.create_long_message_sessions()                                                                                                                 
        self.long_sessions_names = [s.llm_chat_url for s in long_sessions]                                                                                                  
        self.long_sessions: list[list] = [                                                                                                                                  
            [s, INITIAL_TOKENS] for s in long_sessions                                                                                                                      
        ]                                                                                                                                                                   
        small_sessions = self.create_small_message_sessions()                                                                                                               
        self.small_sessions_names = [s.llm_chat_url for s in small_sessions]                                                                                                
        self.small_sessions: list[list] = [                                                                                                                                 
            [s, INITIAL_TOKENS] for s in small_sessions                                                                                                                     
        ]                                                                                                                                                                   
        # FIFO queue of (session, release_time, pool_name) — pool_name is "long" or "small"                                                                                 
        self.timeout_queue: deque[tuple[LLMSession, float, str]] = deque()                                                                                                  
                                                                                                                                                                            
    def _release_timed_out_sessions(self):                                                                                                                                  
        """Move sessions whose timeout has expired back to their original pool."""                                                                                          
        now = time()                                                                                                                                                        
        while self.timeout_queue and self.timeout_queue[0][1] <= now:                                                                                                       
            session, _, pool_name = self.timeout_queue.popleft()                                                                                                            
            pool = self.long_sessions if pool_name == "long" else self.small_sessions                                                                                       
            pool.append([session, INITIAL_TOKENS])                                                                                                                          
                                                                                                                                                                            
    def _move_to_timeout(self, entry: list, pool: list[list]):                                                                                                              
        """Remove entry from its pool and enqueue it for TIMEOUT_HOURS."""                                                                                                  
        pool_name = "long" if pool is self.long_sessions else "small"                                                                                                       
        pool.remove(entry)                                                                                                                                                  
        self.timeout_queue.append((entry[0], time() + TIMEOUT_HOURS * 3600, pool_name))                                                                                     
                                                                                                                                                                            
    def _try_sessions(self, pool: list[list], message: str) -> str:                                                                                                         
        """                                                                                                                                                                 
        Try each session in pool in order. On success rotate it to the end.                                                                                                 
        On failure apply penalty; if tokens < 0 move to timeout queue.                                                                                                      
        Raises RuntimeError if all sessions fail.                                                                                                                           
        """                                                                                                                                                                 
        for entry in list(pool):                                                                                                                                            
            session, _ = entry                                                                                                                                              
            try:                                                                                                                                                            
                logger.info(f"Trying {type(session).__name__} ...")                                                                                                               
                answer = session.send_message(message)   
                pool.remove(entry)                                                                                                                                          
                pool.append(entry)                                                                                                                                          
                return answer                                                                                                                                               
            except Exception:                                                                                                                                               
                entry[1] -= PENALTY_TOKENS                                                                                                                                  
                if entry[1] < 0:                                                                                                                                            
                    self._move_to_timeout(entry, pool)                                                                                                                      
                                                                                                                                                                            
        raise RuntimeError("All LLM sessions failed.")                                                                                                                      
                                                                                                                                                                            
    def call(self, message: str) -> str:                                                                                                                                    
        self._release_timed_out_sessions()
                                                                                                                                                                            
        if len(message) < SMALL_CONTEXT_THRESHOLD and self.small_sessions:                                                                                                  
            try:                                                                                                                                                            
                return self._try_sessions(self.small_sessions, message)                                                                                                     
            except RuntimeError:                                                                                                                                            
                pass  # fall through to long sessions                                                                                                                       
                                                                                                                                                                            
        return self._try_sessions(self.long_sessions, message)
