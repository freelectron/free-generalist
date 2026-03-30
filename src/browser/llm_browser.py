from datetime import datetime
import os
from collections import deque
from time import time         

import tiktoken
import mlflow

from browser.llms.session import LLMSession
from browser.llms.session import ChatGPT, DeepSeek, Gemini, Qwen, Claude, Mistral                                                                                           
from browser.browser import ChromeBrowser
from clog import get_logger
                                                                                                                                                                            
                                                                                                                                                                            
SMALL_CONTEXT_THRESHOLD = 32000                                                                                                                                             
INITIAL_TOKENS = 3
PENALTY_TOKENS = 1
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
        return [self.CLAUDE_SESSION,self.DEEPSEEK_SESSION,  self.MISTRAL_SESSION, self.QWEN_SESSION]
                                                                                                                                                                            
    def create_small_message_sessions(self) -> list[LLMSession]:                                                                                                            
        """Sessions that can only (easily) handle a low number of tokens."""                                                                                                
        return [self.GEMINI_SESSION, self.CHATGPT_SESSION]     

    def record_token_count_in(self, text:str):
        n_tokens = len(self.estimattion_n_token_encoding.encode(text))
        # Keep track of tokens
        mlflow.log_metric( self.tokens_count_in_metric_name, n_tokens, step=self.n_call)
    
    def record_token_count_out(self, text:str):
        n_tokens = len(self.estimattion_n_token_encoding.encode(text))
        # Keep track of tokens
        mlflow.log_metric( self.tokens_count_out_metric_name, n_tokens, step=self.n_call)

    def __init__(self):
        # TODO: make it nicer
        assert os.getenv("CHROME_USER_DATA_DIR")
        chrome_browser = ChromeBrowser()

        # Monitoring
        self.log_experiment_name =  type(self).__name__
        self.log_run_name = datetime.now().date().isoformat()
        mlflow.set_experiment(self.log_experiment_name)
        mlflow.start_run(run_name=f"{self.log_run_name}")
        self.tokens_count_in_metric_name = "tokens_in_estimate"
        self.tokens_count_out_metric_name = "tokens_out_estimate"
        self.estimattion_n_token_encoding = tiktoken.encoding_for_model("gpt-4o")                   
        # Number of calls to llms we made                                                                                                                                                 
        self.n_call = 0 

        self.CHATGPT_SESSION = ChatGPT(chrome_browser, session_id="closed_ai")                                                                                                           
        self.DEEPSEEK_SESSION = DeepSeek(chrome_browser, session_id="deep_seek")                                                                                                         
        self.GEMINI_SESSION = Gemini(chrome_browser, session_id="gemini")                                                                                                                
        self.QWEN_SESSION = Qwen(chrome_browser, session_id="qwen")                                                                                                                      
        self.CLAUDE_SESSION = Claude(chrome_browser, session_id="claude")                                                                                                                
        self.MISTRAL_SESSION = Mistral(chrome_browser, session_id="mistral")      

        # list of [session, tokens] — mutable so token credit updates in-place                                                                                              
        long_sessions = self.create_long_message_sessions()                                                                                                                 
        self.long_sessions: list[list] = [                                                                                                                                  
            [s, INITIAL_TOKENS] for s in long_sessions                                                                                                                      
        ]                                                                                                                                                                   
        small_sessions = self.create_small_message_sessions()                                                                                                               
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
            session, token_credit = entry
            try:                                                                                                                                                            
                logger.info(f"Trying {type(session).__name__} ...")                                                                                                               
                answer = session.send_message(message)   
                pool.remove(entry)                                                                                                                                          
                pool.append(entry)

                # For monitoring
                self.n_call += 1
                self.record_token_count_in(message)
                self.record_token_count_out(answer)
                return answer                                                                                                                                               
            except Exception as e:
                logger.error(f"Error occurred when running {session} with token credit {token_credit}: {e}")
                entry[1] -= PENALTY_TOKENS                                                                                                                                  
                if entry[1] < 0:                                                                                                                                            
                    self._move_to_timeout(entry, pool)                                                                                                                      
                                                                                                                                                                            
        raise RuntimeError("All LLM sessions failed.")                                                                                                                      
                                                                                                                                                                            
    def call(self, message: str) -> str:
        # with mlflow.start_run(self.log_run_name):
            self._release_timed_out_sessions()

            logger.info(f"Queue for calling LLM's with long message window: {[(type(sesh).__name__,tk_credit) for sesh,tk_credit in self.long_sessions]}")

            if len(message) < SMALL_CONTEXT_THRESHOLD and self.small_sessions:
                return self._try_sessions(self.small_sessions, message)

            return self._try_sessions(self.long_sessions, message)


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    mlflow.set_experiment("LLMBrowser")

    with mlflow.start_run(run_name="this"):
        LLM_BROWSER = LLMBrowser()

        LLM_BROWSER.call("hello")
        LLM_BROWSER.call("ola")

