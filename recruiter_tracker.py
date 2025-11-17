"""
Recruiter information tracker for extracting and storing job opportunity details.
"""
import json
import re
import ollama


class RecruiterInfoTracker:
    """
    Tracks information about the recruiter/interviewer and job opportunity.
    """
    def __init__(self, applicant_name=None):
        self.recruiter_name = None
        self.contact_info = None
        self.job_synopsis = None
        self.benefits = None
        self.salary = None
        self._conversation_text = ""
        self.applicant_name = applicant_name  # Store applicant name to filter it out
    
    def add_conversation(self, user_message, bot_response):
        """Add a conversation exchange to track."""
        # Note: User is the recruiter/interviewer, Bot is the applicant
        self._conversation_text += f"Recruiter/Interviewer: {user_message}\nApplicant: {bot_response}\n\n"
    
    def extract_info(self, llm_model):
        """
        Uses LLM to extract recruiter and job information from conversation.
        Returns a dict with extracted information.
        """
        if not self._conversation_text.strip():
            return {}
        
        # Build prompt with applicant name exclusion if known
        applicant_exclusion = ""
        if self.applicant_name:
            applicant_exclusion = f"IMPORTANT: The applicant's name is '{self.applicant_name}'. Do NOT extract this name. Only extract the recruiter/interviewer's name (the person talking TO the applicant). "
        
        extraction_prompt = (
            "Extract the following information from this conversation between a job applicant and a recruiter/interviewer. "
            f"{applicant_exclusion}"
            "CRITICAL: Only extract information that is EXPLICITLY MENTIONED in the conversation. "
            "Do NOT infer, assume, or add information that was not directly stated. "
            "Do NOT add common benefits or standard information unless it was specifically mentioned. "
            "If information is not explicitly mentioned, you MUST return 'Not mentioned' for that field. "
            "\n"
            "Extract the name of the RECRUITER/INTERVIEWER (the person talking TO the applicant), NOT the applicant's name. "
            "Only extract the name of the person representing the company who is conducting the interview. "
            "\n"
            "For benefits: Only list benefits that were EXPLICITLY MENTIONED in the conversation. "
            "Do NOT add standard benefits like 'health insurance' or '401k' unless they were specifically discussed. "
            "If no benefits were mentioned, return 'Not mentioned'. "
            "\n"
            "Return ONLY a JSON object with these exact keys:\n"
            "{\n"
            '  "recruiter_name": "name of the recruiter/interviewer (NOT the applicant) or Not mentioned",\n'
            '  "contact_info": "email/phone of the recruiter/interviewer or Not mentioned",\n'
            '  "job_synopsis": "brief job description or Not mentioned",\n'
            '  "benefits": "ONLY benefits explicitly mentioned in conversation or Not mentioned",\n'
            '  "salary": "salary/compensation explicitly mentioned or Not mentioned"\n'
            "}\n\n"
            f"Conversation:\n{self._conversation_text}\n\n"
            "Remember: Only extract what was EXPLICITLY STATED. Do not infer or add information. "
            "JSON only, no other text:"
        )
        
        try:
            response = ollama.generate(
                model=llm_model,
                prompt=extraction_prompt,
                stream=False
            )
            response_text = response['response'].strip()
            
            # Try to extract JSON from response (handle nested braces)
            # Find the first { and match to the last }
            start_idx = response_text.find('{')
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i in range(start_idx, len(response_text)):
                    if response_text[i] == '{':
                        brace_count += 1
                    elif response_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                
                if end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    try:
                        extracted = json.loads(json_str)
                    except json.JSONDecodeError:
                        extracted = {}
                else:
                    extracted = {}
            else:
                extracted = {}
            
            if extracted:
                
                # Update fields if new information is found
                if extracted.get('recruiter_name') and extracted['recruiter_name'] != 'Not mentioned':
                    # Filter out applicant's name if it matches
                    extracted_name = extracted['recruiter_name']
                    if self.applicant_name and extracted_name.lower() == self.applicant_name.lower():
                        # Skip - this is the applicant's name, not the recruiter's
                        pass
                    elif not self.recruiter_name:
                        self.recruiter_name = extracted_name
                
                if extracted.get('contact_info') and extracted['contact_info'] != 'Not mentioned':
                    if not self.contact_info:
                        self.contact_info = extracted['contact_info']
                
                if extracted.get('job_synopsis') and extracted['job_synopsis'] != 'Not mentioned':
                    if not self.job_synopsis:
                        self.job_synopsis = extracted['job_synopsis']
                
                if extracted.get('benefits') and extracted['benefits'] != 'Not mentioned':
                    # Validate that benefits were actually mentioned in conversation
                    benefits_text = extracted['benefits'].lower()
                    conversation_lower = self._conversation_text.lower()
                    # Check if any key benefit words from extraction appear in conversation
                    # This is a simple validation - if benefits contain words not in conversation, be cautious
                    benefit_words = set(benefits_text.split())
                    conversation_words = set(conversation_lower.split())
                    # Allow some common connecting words
                    common_words = {'and', 'or', 'the', 'a', 'an', 'with', 'including', 'plus', 'also'}
                    benefit_keywords = benefit_words - common_words
                    # If most benefit keywords appear in conversation, accept it
                    if benefit_keywords:
                        matches = sum(1 for word in benefit_keywords if word in conversation_words)
                        # Require at least 50% of keywords to match, or if it's a short phrase, all must match
                        if matches >= len(benefit_keywords) * 0.5 or len(benefit_keywords) <= 3:
                            if not self.benefits:
                                self.benefits = extracted['benefits']
                        # If validation fails, don't update benefits
                
                if extracted.get('salary') and extracted['salary'] != 'Not mentioned':
                    if not self.salary:
                        self.salary = extracted['salary']
                
                return extracted
        except Exception as e:
            # Silently fail - extraction is optional
            pass
        
        return {}
    
    def format_for_log(self):
        """Format the tracked information for log file header."""
        lines = []
        lines.append("RECRUITER/INTERVIEWER INFORMATION:")
        lines.append("-" * 80)
        lines.append(f"Name: {self.recruiter_name or 'Not yet mentioned'}")
        lines.append(f"Contact Info: {self.contact_info or 'Not yet mentioned'}")
        lines.append("")
        lines.append("JOB OPPORTUNITY:")
        lines.append("-" * 80)
        lines.append(f"Job Synopsis: {self.job_synopsis or 'Not yet mentioned'}")
        lines.append(f"Benefits: {self.benefits or 'Not yet mentioned'}")
        lines.append(f"Salary: {self.salary or 'Not yet mentioned'}")
        lines.append("")
        lines.append("=" * 80)
        lines.append("")
        return "\n".join(lines)
    
    def has_any_info(self):
        """Check if any information has been extracted."""
        return any([self.recruiter_name, self.contact_info, self.job_synopsis, 
                   self.benefits, self.salary])

