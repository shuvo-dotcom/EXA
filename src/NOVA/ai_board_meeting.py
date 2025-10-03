"""AI Board Meeting System
============================
A system for conducting AI-powered board meetings with specialized C-suite executives.
Each executive has their own personality, expertise, and decision-making approach.

Usage:
    meeting = BoardMeeting()
    meeting.add_participant(CommunicationsOfficer())
    meeting.add_participant(FinanceOfficer())
    meeting.start_meeting("Q3 Budget Review")
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Protocol

try:
    from src.ai.llm_calls.open_ai_calls import run_open_ai_ns as roains
except ImportError:
    roains = None

# Model configuration
base_model = True
prod_model = False
pro_model = False

if base_model == True:
    base_model = "gpt-5-nano"
if prod_model == True:
    prod_model = "gpt-5-mini"
if pro_model == True:
    pro_model = "gpt-5-pro"

class ExecutiveRole(Enum):
    """Enumeration of available C-suite and board roles"""
    CHAIR = "Chair"
    CEO = "Chief Executive Officer"
    CFO = "Chief Financial Officer"
    CTO = "Chief Technology Officer"
    CMO = "Chief Marketing Officer"
    COO = "Chief Operating Officer"
    CHRO = "Chief Human Resources Officer"
    CLO = "Chief Legal Officer"
    CIO = "Chief Information Officer"
    CCO = "Chief Communications Officer"
    EXECUTIVE_ASSISTANT = "Executive Assistant"
    BOARD_MEMBER = "Board Member"


class MeetingPhase(Enum):
    """Different phases of a board meeting"""
    OPENING = "opening"
    AGENDA_REVIEW = "agenda_review"
    DISCUSSION = "discussion"
    DECISION_MAKING = "decision_making"
    ACTION_ITEMS = "action_items"
    CLOSING = "closing"


@dataclass
class MeetingContext:
    """Context information for the current meeting"""
    topic: str
    phase: MeetingPhase
    agenda_items: List[str] = field(default_factory=list)
    decisions_made: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    meeting_notes: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)


class BoardMemberProtocol(Protocol):
    """Protocol defining the interface for board members"""
    def respond_to_topic(self, topic: str, context: MeetingContext) -> str: ...
    def provide_expertise(self, question: str, context: MeetingContext) -> str: ...
    def vote_on_proposal(self, proposal: str, context: MeetingContext) -> tuple[str, str]: ...


class BaseBoardMember(ABC):
    """Abstract base class for all board members"""
    
    def __init__(self, name: str, role: ExecutiveRole, personality_traits: List[str] = None):
        self.member_id = str(uuid.uuid4())
        self.name = name
        self.role = role
        self.personality_traits = personality_traits or []
        self.expertise_areas = self._define_expertise()
        self.communication_style = self._define_communication_style()
        self.decision_framework = self._define_decision_framework()
    
    @abstractmethod
    def _define_expertise(self) -> List[str]:
        """Define the specific expertise areas for this role"""
        pass
    
    @abstractmethod
    def _define_communication_style(self) -> Dict[str, str]:
        """Define how this executive communicates"""
        pass
    
    @abstractmethod
    def _define_decision_framework(self) -> Dict[str, Any]:
        """Define how this executive makes decisions"""
        pass
    
    def get_system_prompt(self) -> str:
        """Generate the system prompt for this board member"""
        return f"""You are {self.name}, the {self.role.value} in this board meeting.

                    PERSONALITY TRAITS: {', '.join(self.personality_traits)}

                    EXPERTISE AREAS: {', '.join(self.expertise_areas)}

                    COMMUNICATION STYLE:
                    {json.dumps(self.communication_style, indent=2)}

                    DECISION FRAMEWORK:
                    {json.dumps(self.decision_framework, indent=2)}

                    MEETING PROTOCOL:
                    - Only speak when directly called upon or when you have something critical to add
                    - If you want to contribute to a discussion, indicate you want to speak but wait to be called on
                    - Keep responses concise and focused (2-3 sentences max unless specifically asked for detailed analysis)
                    - Stay in character and focus on your areas of expertise
                    - Always write in prose as your text will be spoken out loud
                    - Be collaborative but assert your professional perspective when needed
                    - Address the Chair respectfully and acknowledge other participants' contributions
            """
    
    @abstractmethod
    def respond_to_topic(self, topic: str, context: MeetingContext) -> str:
        """Respond to a meeting topic from this executive's perspective"""
        pass
    
    @abstractmethod
    def provide_expertise(self, question: str, context: MeetingContext) -> str:
        """Provide expert advice on a specific question"""
        pass
    
    @abstractmethod
    def vote_on_proposal(self, proposal: str, context: MeetingContext) -> tuple[str, str]:
        """Vote on a proposal and provide reasoning"""
        pass
    
    def wants_to_speak(self, topic: str, context: MeetingContext) -> tuple[bool, str]:
        """Determine if this member wants to contribute to the discussion"""
        if roains:
            prompt = f"Given the discussion topic '{topic}', do you have something important to contribute from your role as {self.role.value}? Respond with 'YES' if you want to speak, 'NO' if you don't need to contribute. If YES, briefly indicate why (one sentence)."
            response = roains(prompt, self.get_system_prompt(), model=base_model)
            
            if response.upper().startswith('YES'):
                # Extract the reason (everything after YES)
                reason = response[3:].strip().lstrip(',').lstrip(':').strip()
                return True, reason
            else:
                return False, ""
        else:
            # Fallback: everyone wants to speak
            return True, f"I have insights from my {self.role.value} perspective."


class HumanChair:
    """Human Chair - represents the human meeting facilitator"""
    
    def __init__(self, name: str = "Meeting Chair"):
        self.member_id = str(uuid.uuid4())
        self.name = name
        self.role = ExecutiveRole.CHAIR
        self.is_human = True
    
    def add_comment(self, comment: str) -> str:
        """Add a comment from the human chair"""
        return comment
    
    def vote_on_proposal(self, vote: str, reasoning: str = "") -> tuple[str, str]:
        """Human chair votes manually"""
        return (vote, reasoning)


class AIChair(BaseBoardMember):
    """AI Meeting Chair - alternative to human chair if needed"""
    
    def __init__(self, name: str = "AI Board Chair"):
        super().__init__(name, ExecutiveRole.CHAIR, 
                        ["diplomatic", "decisive", "strategic", "collaborative"])
    
    def _define_expertise(self) -> List[str]:
        return ["corporate governance", "strategic planning", "stakeholder management", 
                "meeting facilitation", "organizational leadership"]
    
    def _define_communication_style(self) -> Dict[str, str]:
        return {
            "tone": "authoritative yet inclusive",
            "approach": "facilitates discussion, summarizes key points",
            "focus": "keeps meetings on track and ensures all voices are heard"
        }
    
    def _define_decision_framework(self) -> Dict[str, Any]:
        return {
            "primary_considerations": ["stakeholder impact", "strategic alignment", "governance compliance"],
            "decision_style": "consensus-building with final authority",
            "risk_tolerance": "moderate"
        }
    
    def respond_to_topic(self, topic: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As the Board Chair, provide opening remarks for discussion topic: '{topic}'. Set the context and guide the discussion."
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return f"Thank you all for joining. Let's discuss {topic} and ensure we have input from all relevant perspectives."
    
    def provide_expertise(self, question: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As Chair, provide governance guidance on: {question}"
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return "From a governance perspective, we need to ensure this aligns with our fiduciary responsibilities."
    
    def vote_on_proposal(self, proposal: str, context: MeetingContext) -> tuple[str, str]:
        # Chair typically votes last and may break ties
        return ("APPROVE", "After considering all perspectives, this appears to be in the best interest of the organization.")


class CommunicationsOfficer(BaseBoardMember):
    """Chief Communications Officer - handles messaging, PR, and stakeholder communications"""
    
    def __init__(self, name: str = "Lola"):
        super().__init__(name, ExecutiveRole.CCO, 
                        ["articulate", "strategic", "brand-conscious", "media-savvy"])
    
    def _define_expertise(self) -> List[str]:
        return ["public relations", "crisis communications", "brand management", 
                "stakeholder engagement", "media strategy", "internal communications"]
    
    def _define_communication_style(self) -> Dict[str, str]:
        return {
            "tone": "polished and strategic",
            "approach": "considers messaging implications of all decisions",
            "focus": "brand reputation and stakeholder perception"
        }
    
    def _define_decision_framework(self) -> Dict[str, Any]:
        return {
            "primary_considerations": ["reputational impact", "stakeholder messaging", "brand alignment"],
            "decision_style": "collaborative with strong advocacy for communication aspects",
            "risk_tolerance": "low for reputational risks"
        }
    
    def respond_to_topic(self, topic: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As CCO, analyze the communications implications of: '{topic}'. Consider stakeholder messaging and brand impact."
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return f"From a communications perspective, we need to consider how {topic} will be perceived by our stakeholders."
    
    def provide_expertise(self, question: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As CCO, provide communications strategy advice on: {question}"
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return "We need to develop a clear messaging strategy that aligns with our brand values."
    
    def vote_on_proposal(self, proposal: str, context: MeetingContext) -> tuple[str, str]:
        return ("CONDITIONAL", "I support this pending development of appropriate stakeholder communications.")


class FinanceOfficer(BaseBoardMember):
    """Chief Financial Officer - handles financial analysis, budgets, and fiscal responsibility"""
    
    def __init__(self, name: str = "Sarah"):
        super().__init__(name, ExecutiveRole.CFO, 
                        ["analytical", "detail-oriented", "risk-aware", "data-driven"])
    
    def _define_expertise(self) -> List[str]:
        return ["financial analysis", "budgeting", "risk management", "investor relations", 
                "regulatory compliance", "cost optimization", "financial modeling"]
    
    def _define_communication_style(self) -> Dict[str, str]:
        return {
            "tone": "precise and fact-based",
            "approach": "data-driven analysis with clear financial implications",
            "focus": "financial sustainability and ROI"
        }
    
    def _define_decision_framework(self) -> Dict[str, Any]:
        return {
            "primary_considerations": ["financial impact", "ROI", "budget implications", "cash flow"],
            "decision_style": "evidence-based with conservative bias",
            "risk_tolerance": "low to moderate"
        }
    
    def respond_to_topic(self, topic: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As CFO, analyze the financial implications of: '{topic}'. Provide budget impact and ROI considerations."
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return f"I need to analyze the financial impact of {topic} on our budget and cash flow projections."
    
    def provide_expertise(self, question: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As CFO, provide financial analysis and recommendations for: {question}"
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return "Let me provide the financial analysis and budget implications for this decision."
    
    def vote_on_proposal(self, proposal: str, context: MeetingContext) -> tuple[str, str]:
        return ("CONDITIONAL", "I support this if we can identify funding sources and maintain fiscal responsibility.")


class TechnologyOfficer(BaseBoardMember):
    """Chief Technology Officer - handles technology strategy, innovation, and digital transformation"""
    
    def __init__(self, name: str = "Marcus Kim"):
        super().__init__(name, ExecutiveRole.CTO, 
                        ["innovative", "technical", "forward-thinking", "systems-oriented"])
    
    def _define_expertise(self) -> List[str]:
        return ["technology strategy", "digital transformation", "cybersecurity", "innovation management", 
                "system architecture", "emerging technologies", "technical due diligence"]
    
    def _define_communication_style(self) -> Dict[str, str]:
        return {
            "tone": "technical but accessible",
            "approach": "solutions-oriented with focus on scalability",
            "focus": "technological feasibility and innovation opportunities"
        }
    
    def _define_decision_framework(self) -> Dict[str, Any]:
        return {
            "primary_considerations": ["technical feasibility", "scalability", "security", "innovation potential"],
            "decision_style": "methodical with emphasis on long-term sustainability",
            "risk_tolerance": "moderate to high for technological innovation"
        }
    
    def respond_to_topic(self, topic: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As CTO, analyze the technology implications of: '{topic}'. Consider technical feasibility and innovation opportunities."
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return f"From a technology perspective, we need to evaluate the technical requirements and scalability of {topic}."
    
    def provide_expertise(self, question: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As CTO, provide technical guidance and recommendations for: {question}"
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return "Let me address the technical aspects and provide implementation recommendations."
    
    def vote_on_proposal(self, proposal: str, context: MeetingContext) -> tuple[str, str]:
        return ("APPROVE", "This aligns with our technology roadmap and innovation objectives.")


class LegalOfficer(BaseBoardMember):
    """Chief Legal Officer - handles legal compliance, risk assessment, and regulatory matters"""
    
    def __init__(self, name: str = "Dr. Jennifer Walsh"):
        super().__init__(name, ExecutiveRole.CLO, 
                        ["cautious", "thorough", "compliance-focused", "risk-aware"])
    
    def _define_expertise(self) -> List[str]:
        return ["corporate law", "regulatory compliance", "contract negotiation", "risk assessment", 
                "intellectual property", "litigation management", "governance"]
    
    def _define_communication_style(self) -> Dict[str, str]:
        return {
            "tone": "precise and careful",
            "approach": "thorough risk analysis with compliance focus",
            "focus": "legal implications and regulatory requirements"
        }
    
    def _define_decision_framework(self) -> Dict[str, Any]:
        return {
            "primary_considerations": ["legal compliance", "regulatory requirements", "liability exposure"],
            "decision_style": "conservative with emphasis on risk mitigation",
            "risk_tolerance": "very low"
        }
    
    def respond_to_topic(self, topic: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As CLO, analyze the legal and compliance implications of: '{topic}'. Identify potential risks and regulatory requirements."
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return f"I need to review the legal implications and compliance requirements for {topic}."
    
    def provide_expertise(self, question: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As CLO, provide legal guidance and risk assessment for: {question}"
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return "From a legal standpoint, we need to ensure full compliance with applicable regulations."
    
    def vote_on_proposal(self, proposal: str, context: MeetingContext) -> tuple[str, str]:
        return ("CONDITIONAL", "I support this pending legal review and appropriate risk mitigation measures.")


class InformationOfficer(BaseBoardMember):
    """Chief Information Officer - handles information systems, data management, and IT operations"""
    
    def __init__(self, name: str = "Ivan"):
        super().__init__(name, ExecutiveRole.CIO, 
                        ["systematic", "data-driven", "security-conscious", "efficiency-focused"])
    
    def _define_expertise(self) -> List[str]:
        return ["information systems", "data management", "IT operations", "cybersecurity", 
                "digital infrastructure", "business intelligence", "IT governance"]
    
    def _define_communication_style(self) -> Dict[str, str]:
        return {
            "tone": "logical and systematic",
            "approach": "data-informed decisions with operational focus",
            "focus": "information security and operational efficiency"
        }
    
    def _define_decision_framework(self) -> Dict[str, Any]:
        return {
            "primary_considerations": ["data security", "operational efficiency", "system reliability"],
            "decision_style": "systematic with emphasis on security and stability",
            "risk_tolerance": "low for security risks, moderate for operational changes"
        }
    
    def respond_to_topic(self, topic: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As CIO, analyze the information systems and data implications of: '{topic}'. Consider security and operational impact."
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return f"I need to assess the information systems and data security implications of {topic}."
    
    def provide_expertise(self, question: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As CIO, provide information systems guidance for: {question}"
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return "Let me address the data management and IT infrastructure requirements."
    
    def vote_on_proposal(self, proposal: str, context: MeetingContext) -> tuple[str, str]:
        return ("CONDITIONAL", "I support this with appropriate data governance and security measures in place.")


class ExecutiveAssistant(BaseBoardMember):
    """Executive Assistant - supports meeting coordination and administrative functions"""
    
    def __init__(self, name: str = "Emma Thompson"):
        super().__init__(name, ExecutiveRole.EXECUTIVE_ASSISTANT, 
                        ["organized", "supportive", "detail-oriented", "diplomatic"])
    
    def _define_expertise(self) -> List[str]:
        return ["meeting coordination", "administrative support", "stakeholder liaison", 
                "document management", "scheduling", "protocol management"]
    
    def _define_communication_style(self) -> Dict[str, str]:
        return {
            "tone": "professional and supportive",
            "approach": "facilitates smooth operations and clear communication",
            "focus": "ensuring all participants have necessary information and support"
        }
    
    def _define_decision_framework(self) -> Dict[str, Any]:
        return {
            "primary_considerations": ["operational efficiency", "stakeholder needs", "process optimization"],
            "decision_style": "supportive with focus on implementation",
            "risk_tolerance": "moderate"
        }
    
    def respond_to_topic(self, topic: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As Executive Assistant, provide administrative and coordination perspective on: '{topic}'. Focus on implementation support."
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return f"I can coordinate the administrative aspects and ensure proper documentation for {topic}."
    
    def provide_expertise(self, question: str, context: MeetingContext) -> str:
        if roains:
            prompt = f"As Executive Assistant, provide coordination and administrative guidance for: {question}"
            return roains(prompt, self.get_system_prompt(), model=base_model)
        return "I can help coordinate the implementation and ensure all stakeholders are properly informed."
    
    def vote_on_proposal(self, proposal: str, context: MeetingContext) -> tuple[str, str]:
        return ("SUPPORT", "I'll ensure proper documentation and coordinate implementation if approved.")


class BoardMeeting:
    """Main class for managing AI board meetings with human chair"""
    
    def __init__(self, chair_name: str = "Meeting Chair", use_ai_chair: bool = False):
        self.meeting_id = str(uuid.uuid4())
        self.participants: Dict[str, BaseBoardMember] = {}
        self.context = MeetingContext("", MeetingPhase.OPENING)
        
        # Create chair based on preference
        if use_ai_chair:
            self.chair = AIChair(chair_name)
            self.participants[self.chair.member_id] = self.chair
        else:
            self.chair = HumanChair(chair_name)
            # Human chair is not added to participants since they're not AI
            
        self.current_topic = ""
        self.raised_hands: List[tuple[str, str]] = []  # (member_name, reason)
        self.chat_log: List[Dict[str, str]] = []  # Chat history
        
    def add_chat_message(self, speaker: str, message: str, message_type: str = "speech"):
        """Add a message to the chat log"""
        self.chat_log.append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "speaker": speaker,
            "message": message,
            "type": message_type
        })
        
    def display_chat_message(self, speaker: str, message: str, message_type: str = "speech"):
        """Display a chat message in a formatted way"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if message_type == "system":
            print(f"\n🔔 [{timestamp}] SYSTEM: {message}")
        elif message_type == "chair":
            print(f"\n👑 [{timestamp}] {speaker}: {message}")
        else:
            # Find the role emoji
            role_emoji = "👤"
            for member in self.participants.values():
                if member.name == speaker:
                    role_emoji = {
                        ExecutiveRole.CCO: "📢",
                        ExecutiveRole.CFO: "💰", 
                        ExecutiveRole.CTO: "💻",
                        ExecutiveRole.CLO: "⚖️",
                        ExecutiveRole.CIO: "📊",
                        ExecutiveRole.EXECUTIVE_ASSISTANT: "📝"
                    }.get(member.role, "👤")
                    break
            print(f"\n{role_emoji} [{timestamp}] {speaker}: {message}")
        
        self.add_chat_message(speaker, message, message_type)
        
    def add_participant(self, member: BaseBoardMember):
        """Add a board member to the meeting"""
        self.participants[member.member_id] = member
        
    def remove_participant(self, member_id: str):
        """Remove a board member from the meeting"""
        if member_id in self.participants:
            # Can't remove human chair, but can remove AI chair
            if hasattr(self.chair, 'is_human') and self.chair.is_human:
                del self.participants[member_id]
            elif member_id != self.chair.member_id:
                del self.participants[member_id]
            
    def get_participants_summary(self) -> str:
        """Get a summary of all meeting participants"""
        total_count = len(self.participants)
        if hasattr(self.chair, 'is_human') and self.chair.is_human:
            total_count += 1  # Add human chair to count
            
        summary = f"Meeting Participants ({total_count}):\n"
        
        # Add chair info
        summary += f"- {self.chair.name} ({self.chair.role.value}) {'[HUMAN]' if hasattr(self.chair, 'is_human') and self.chair.is_human else '[AI]'}\n"
        
        # Add other participants
        for member in self.participants.values():
            if not (hasattr(self.chair, 'is_human') and self.chair.is_human) and member == self.chair:
                continue  # Skip AI chair as it's already added above
            summary += f"- {member.name} ({member.role.value}) [AI]\n"
        return summary
        
    def start_meeting(self, topic: str, agenda_items: List[str] = None):
        """Start the board meeting with a specific topic"""
        self.context.topic = topic
        self.context.agenda_items = agenda_items or []
        self.context.phase = MeetingPhase.AGENDA_REVIEW
        
        self.display_chat_message("SYSTEM", "Board meeting started", "system")
        self.display_chat_message("SYSTEM", f"Topic: {topic}", "system")
        
        if agenda_items:
            agenda_text = "\n".join([f"  • {item}" for item in agenda_items])
            self.display_chat_message("SYSTEM", f"Agenda:\n{agenda_text}", "system")
        
        # If AI chair, give opening remarks. If human chair, they'll provide their own.
        if hasattr(self.chair, 'is_human') and self.chair.is_human:
            self.display_chat_message("SYSTEM", f"Waiting for {self.chair.name} to provide opening remarks...", "system")
        else:
            # AI Chair opens the meeting
            opening_remarks = self.chair.respond_to_topic(topic, self.context)
            self.display_chat_message(self.chair.name, opening_remarks, "chair")
        
    def introduce_topic(self, topic: str) -> List[tuple[str, str]]:
        """Introduce a new topic and get list of who wants to speak"""
        self.current_topic = topic
        self.raised_hands = []
        
        self.display_chat_message("SYSTEM", f"New topic introduced: {topic}", "system")
        
        # Check who wants to speak (excluding chair if human)
        for member in self.participants.values():
            if hasattr(self.chair, 'is_human') and self.chair.is_human:
                # For human chair, all AI members may want to speak
                wants_speak, reason = member.wants_to_speak(topic, self.context)
                if wants_speak:
                    self.raised_hands.append((member.name, reason))
            else:
                # For AI chair, exclude chair from wanting to speak
                if member != self.chair:
                    wants_speak, reason = member.wants_to_speak(topic, self.context)
                    if wants_speak:
                        self.raised_hands.append((member.name, reason))
                    
        if self.raised_hands:
            hands_text = "\n".join([f"  🙋 {name}: {reason}" for name, reason in self.raised_hands])
            self.display_chat_message("SYSTEM", f"Raised hands:\n{hands_text}", "system")
        else:
            self.display_chat_message("SYSTEM", "No one has raised their hand to speak.", "system")
            
        return self.raised_hands
        
    def call_on_speaker(self, speaker_name: str) -> str:
        """Call on a specific speaker to contribute"""
        # Find the member
        speaker_member = None
        for member in self.participants.values():
            if member.name == speaker_name:
                speaker_member = member
                break
                
        if not speaker_member:
            self.display_chat_message("SYSTEM", f"Speaker {speaker_name} not found.", "system")
            return ""
            
        # Remove from raised hands if present
        self.raised_hands = [(name, reason) for name, reason in self.raised_hands if name != speaker_name]
        
        # Get response
        response = speaker_member.respond_to_topic(self.current_topic, self.context)
        self.display_chat_message(speaker_name, response)
        
        return response
        
    def ask_specific_question(self, speaker_name: str, question: str) -> str:
        """Ask a specific question to a board member"""
        # Find the member
        speaker_member = None
        for member in self.participants.values():
            if member.name == speaker_name:
                speaker_member = member
                break
                
        if not speaker_member:
            self.display_chat_message("SYSTEM", f"Speaker {speaker_name} not found.", "system")
            return ""
            
        self.display_chat_message("SYSTEM", f"Question to {speaker_name}: {question}", "system")
        
        response = speaker_member.provide_expertise(question, self.context)
        self.display_chat_message(speaker_name, response)
        
        return response
        
    def get_available_speakers(self) -> List[str]:
        """Get list of members who have raised their hands"""
        return [name for name, _ in self.raised_hands]
        
    def chair_comment(self, comment: str):
        """Allow the chair to make a comment (human or AI)"""
        if hasattr(self.chair, 'is_human') and self.chair.is_human:
            # Human chair comment
            self.display_chat_message(self.chair.name, comment, "chair")
        else:
            # AI chair comment
            self.display_chat_message(self.chair.name, comment, "chair")
    
    def human_chair_vote(self, vote: str, reasoning: str = "") -> tuple[str, str]:
        """Allow human chair to cast their vote"""
        if hasattr(self.chair, 'is_human') and self.chair.is_human:
            return self.chair.vote_on_proposal(vote, reasoning)
        else:
            raise ValueError("This method is only for human chairs")
        
    def discuss_topic(self, topic: str) -> Dict[str, str]:
        """Legacy method - introduces topic and gets responses from all who want to speak"""
        raised_hands = self.introduce_topic(topic)
        responses = {}
        
        # Get responses from everyone who raised their hand
        for speaker_name, reason in raised_hands:
            response = self.call_on_speaker(speaker_name)
            responses[speaker_name] = response
            
        return responses
        
    def seek_expertise(self, question: str, role: ExecutiveRole = None) -> Dict[str, str]:
        """Seek specific expertise from one or all members"""
        responses = {}
        
        if role:
            # Seek expertise from specific role
            for member in self.participants.values():
                if member.role == role:
                    response = self.ask_specific_question(member.name, question)
                    responses[member.name] = response
        else:
            # Ask all members the question (excluding human chair)
            self.display_chat_message("SYSTEM", f"SEEKING EXPERTISE: {question}", "system")
            for member in self.participants.values():
                # Skip AI chair if it exists, human chair handles themselves
                if not (hasattr(self.chair, 'is_human') and not self.chair.is_human and member == self.chair):
                    response = self.ask_specific_question(member.name, question)
                    responses[member.name] = response
                    
        return responses
        
    def vote_on_proposal(self, proposal: str) -> Dict[str, tuple[str, str]]:
        """Legacy method - calls for vote"""
        return self.call_for_vote(proposal)
        
    def call_for_vote(self, proposal: str) -> Dict[str, tuple[str, str]]:
        """Call for a vote on a proposal - all members must vote (except human chair votes separately)"""
        votes = {}
        self.display_chat_message("SYSTEM", f"VOTING ON PROPOSAL: {proposal}", "system")
        
        for member in self.participants.values():
            vote, reasoning = member.vote_on_proposal(proposal, self.context)
            votes[member.name] = (vote, reasoning)
            
            # Format vote display
            vote_emoji = {"APPROVE": "✅", "REJECT": "❌", "CONDITIONAL": "⚠️", "SUPPORT": "👍"}.get(vote, "🗳️")
            self.display_chat_message(member.name, f"{vote_emoji} {vote}: {reasoning}")
            
        # Note if human chair needs to vote separately
        if hasattr(self.chair, 'is_human') and self.chair.is_human:
            self.display_chat_message("SYSTEM", f"Waiting for {self.chair.name} to cast their vote...", "system")
        
        return votes
        
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the complete chat history"""
        return self.chat_log.copy()
        
    def export_chat_log(self) -> str:
        """Export chat log as formatted text"""
        output = f"=== BOARD MEETING CHAT LOG ===\n"
        output += f"Meeting ID: {self.meeting_id}\n"
        output += f"Topic: {self.context.topic}\n"
        output += f"Started: {self.context.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for entry in self.chat_log:
            if entry["type"] == "system":
                output += f"🔔 [{entry['timestamp']}] SYSTEM: {entry['message']}\n"
            elif entry["type"] == "chair":
                output += f"👑 [{entry['timestamp']}] {entry['speaker']}: {entry['message']}\n"
            else:
                output += f"👤 [{entry['timestamp']}] {entry['speaker']}: {entry['message']}\n"
            output += "\n"
            
        return output
        
    def conclude_meeting(self):
        """Conclude the meeting and generate summary"""
        self.context.phase = MeetingPhase.CLOSING
        print(f"\n=== MEETING CONCLUDED ===")
        print(f"Duration: {datetime.now() - self.context.start_time}")
        print(f"Decisions Made: {len(self.context.decisions_made)}")
        print(f"Action Items: {len(self.context.action_items)}")


# Factory functions for easy instantiation
def create_standard_board(chair_name: str = "Meeting Chair", use_ai_chair: bool = False) -> BoardMeeting:
    """Create a meeting with standard C-suite executives and human chair by default"""
    meeting = BoardMeeting(chair_name=chair_name, use_ai_chair=use_ai_chair)
    meeting.add_participant(CommunicationsOfficer(name = 'Lola'))
    meeting.add_participant(FinanceOfficer(name = 'Sarah'))
    meeting.add_participant(TechnologyOfficer(name = 'Emil'))
    meeting.add_participant(LegalOfficer(name = 'Anna'))
    meeting.add_participant(ExecutiveAssistant(name = 'Nova'))
    return meeting


def create_custom_board(roles: List[ExecutiveRole], names: List[str] = None, 
                       chair_name: str = "Meeting Chair", use_ai_chair: bool = False) -> BoardMeeting:
    """Create a meeting with custom selection of executives and human chair by default"""
    meeting = BoardMeeting(chair_name=chair_name, use_ai_chair=use_ai_chair)
    
    role_map = {
        ExecutiveRole.CCO: CommunicationsOfficer,
        ExecutiveRole.CFO: FinanceOfficer,
        ExecutiveRole.CTO: TechnologyOfficer,
        ExecutiveRole.CLO: LegalOfficer,
        ExecutiveRole.CIO: InformationOfficer,
        ExecutiveRole.EXECUTIVE_ASSISTANT: ExecutiveAssistant,
    }
    
    for i, role in enumerate(roles):
        if role in role_map:
            name = names[i] if names and i < len(names) else None
            if name:
                meeting.add_participant(role_map[role](name))
            else:
                meeting.add_participant(role_map[role]())
                
    return meeting


# if __name__ == "__main__":
#     # Example usage - Controlled Meeting Format
#     # This only runs when the file is executed directly, not when imported
#     print("Creating AI Board Meeting...")
    
#     # Create a standard board meeting
#     meeting = create_standard_board()
    
#     # Start the meeting
#     meeting.start_meeting("Q4 Strategic Planning", 
#                          ["Budget Review", "Technology Roadmap", "Market Expansion"])
    
#     # Introduce a topic and see who wants to speak
#     raised_hands = meeting.introduce_topic("Digital Transformation Initiative")
    
#     # Call on specific speakers
#     if raised_hands:
#         # Call on first person who raised their hand
#         first_speaker = raised_hands[0][0]
#         meeting.call_on_speaker(first_speaker)
        
#         # Chair makes a comment
#         meeting.chair_comment("Thank you for that perspective. Let's hear from finance on the budget implications.")
        
#         # Ask specific question to CFO
#         meeting.ask_specific_question("Sarah Rodriguez", "What's the estimated budget impact of this digital transformation?")
    
#     # Call for vote on a proposal
#     meeting.call_for_vote("Allocate $2M for AI infrastructure upgrades")
    
#     # Export chat log
#     chat_log = meeting.export_chat_log()
#     print("\n" + "="*50)
#     print("FULL CHAT LOG:")
#     print("="*50)
#     print(chat_log)
