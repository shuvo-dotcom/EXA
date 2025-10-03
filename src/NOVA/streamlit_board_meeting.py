"""Streamlit Interface for AI Board Meeting System
===============    if 'quick_proposal_selected' not in st.session_state:
        st.session_state.quick_proposal_selected = False
    if 'is_human_chair' not in st.session_state:
        st.session_state.is_human_chair = True

def create_meeting_interface():===============================
Interactive web         # Additional chair actions
        st.subheader("🔧 Chair Tools")
        if st.button("📋 Meeting Minutes", key="minutes_btn"):
            if hasattr(st.session_state.meeting, 'discussion_history'):
                st.info("Meeting minutes functionality coming soon...")
            else:
                st.warning("No discussion history available.")

def expertise_interface():ducting AI-powered board meetings.

To run:
    streamlit run src/NOVA/streamlit_board_meeting.py

Features:
- Select board members for the meeting
- Set meeting topics and agenda
- Real-time discussion with AI executives
- Vote on proposals
- Export meeting minutes
"""

import streamlit as st
import json
from datetime import datetime
from typing import Dict, List, Optional
import sys
import os

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import board meeting classes
try:
    from src.NOVA.ai_board_meeting import (
        BoardMeeting, 
        ExecutiveRole,
        CommunicationsOfficer,
        FinanceOfficer,
        TechnologyOfficer,
        LegalOfficer,
        InformationOfficer,
        ExecutiveAssistant
    )
except ImportError as e:
    st.error(f"Could not import board meeting modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Board Meeting",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'meeting' not in st.session_state:
        st.session_state.meeting = None
    if 'meeting_started' not in st.session_state:
        st.session_state.meeting_started = False
    if 'discussion_history' not in st.session_state:
        st.session_state.discussion_history = []
    if 'votes_history' not in st.session_state:
        st.session_state.votes_history = []
    if 'quick_topic_selected' not in st.session_state:
        st.session_state.quick_topic_selected = False
    if 'quick_question_selected' not in st.session_state:
        st.session_state.quick_question_selected = False
    if 'quick_proposal_selected' not in st.session_state:
        st.session_state.quick_proposal_selected = False
    if 'is_human_chair' not in st.session_state:
        st.session_state.is_human_chair = True

def create_meeting_interface():
    """Interface for creating a new board meeting"""
    st.header("🏢 AI Board Meeting Setup")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Select Board Members")
        
        # Available executive roles
        available_executives = {
            "Communications Officer (CCO)": CommunicationsOfficer,
            "Finance Officer (CFO)": FinanceOfficer,
            "Technology Officer (CTO)": TechnologyOfficer,
            "Legal Officer (CLO)": LegalOfficer,
            "Information Officer (CIO)": InformationOfficer,
            "Executive Assistant": ExecutiveAssistant
        }
        
        # Multi-select for board members
        selected_members = st.multiselect(
            "Choose executives to participate:",
            list(available_executives.keys()),
            default=["Communications Officer (CCO)", "Finance Officer (CFO)", "Technology Officer (CTO)"],
            key="member_selection"
        )
        
        if len(selected_members) < 2:
            st.warning("⚠️ Please select at least 2 board members for a productive meeting.")
            return
        
        # Chair configuration
        st.subheader("👑 Meeting Chair")
        chair_type = st.radio(
            "Choose chair type:",
            options=["Human Chair (You)", "AI Chair"],
            index=0,
            key="chair_type_selection",
            help="Human Chair: You control the meeting directly. AI Chair: AI facilitates automatically."
        )
        
        if chair_type == "Human Chair (You)":
            chair_name = st.text_input(
                "Your name as chair:",
                value="Meeting Chair",
                key="human_chair_name"
            )
            use_ai_chair = False
        else:
            chair_name = st.text_input(
                "AI Chair name:",
                value="AI Board Chair",
                key="ai_chair_name"
            )
            use_ai_chair = True
        
        # Meeting topic
        st.subheader("📝 Meeting Topic")
        meeting_topic = st.text_input(
            "Enter the main topic for discussion:",
            placeholder="e.g., Q4 Budget Planning, Product Launch Strategy, etc.",
            key="meeting_topic_input"
        )
        
        # Start meeting button
        if st.button("🚀 Start Board Meeting", type="primary", key="start_meeting_btn"):
            if not meeting_topic.strip():
                st.error("Please enter a meeting topic before starting.")
                return
            
            try:
                # Create new meeting with chair configuration
                meeting = BoardMeeting(chair_name=chair_name, use_ai_chair=use_ai_chair)
                
                # Add selected participants
                for member_name in selected_members:
                    executive_class = available_executives[member_name]
                    meeting.add_participant(executive_class())
                
                # Store in session state
                st.session_state.meeting = meeting
                st.session_state.meeting_started = True
                st.session_state.is_human_chair = not use_ai_chair
                
                # Start the meeting
                meeting.start_meeting(meeting_topic.strip())
                
                st.success(f"✅ Meeting started successfully!")
                if not use_ai_chair:
                    st.info("🎯 You are now the meeting chair. Use the controls below to facilitate the discussion.")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error starting meeting: {str(e)}")
    
    with col2:
        st.subheader("ℹ️ Meeting Info")
        st.info("""
        **How it works:**
        
        1. Select board members
        2. Set a meeting topic
        3. Use the discussion interface
        4. Executives raise hands to speak
        5. You control who speaks next
        6. Vote on proposals
        7. Export meeting minutes
        """)
        
        if selected_members:
            st.subheader("👥 Selected Members")
            for member in selected_members:
                role_emoji = {
                    "Communications Officer (CCO)": "📢",
                    "Finance Officer (CFO)": "💰",
                    "Technology Officer (CTO)": "💻",
                    "Legal Officer (CLO)": "⚖️",
                    "Information Officer (CIO)": "📊",
                    "Executive Assistant": "📝"
                }.get(member, "👤")
                st.write(f"{role_emoji} {member}")

def display_meeting_status():
    """Display current meeting status and participants"""
    if not st.session_state.meeting_started:
        return
    
    meeting = st.session_state.meeting
    st.sidebar.subheader("📊 Meeting Status")
    
    # Meeting info
    st.sidebar.info(f"**Topic:** {meeting.context.topic}")
    st.sidebar.info(f"**Started:** {meeting.context.start_time.strftime('%H:%M:%S')}")
    
    # Chair info
    chair_type = "HUMAN" if st.session_state.get('is_human_chair', True) else "AI"
    st.sidebar.subheader(f"👑 Chair ({chair_type})")
    st.sidebar.write(f"� {meeting.chair.name}")
    
    # Participants
    st.sidebar.subheader("👥 AI Board Members")
    for role_name, member in meeting.participants.items():
        role_emoji = {
            ExecutiveRole.CCO: "📢",
            ExecutiveRole.CFO: "💰",
            ExecutiveRole.CTO: "💻",
            ExecutiveRole.CLO: "⚖️",
            ExecutiveRole.CIO: "📊",
            ExecutiveRole.EXECUTIVE_ASSISTANT: "📝"
        }.get(member.role, "👤")
        st.sidebar.write(f"{role_emoji} {member.name}")
    
    # Show who wants to speak
    if hasattr(meeting, 'raised_hands') and meeting.raised_hands:
        st.sidebar.subheader("✋ Wants to Speak")
        for member_name, reason in meeting.raised_hands:
            st.sidebar.write(f"✋ {member_name}: {reason}")

def discussion_interface():
    """Interface for conducting discussions"""
    if not st.session_state.meeting_started:
        return
    
    st.header("💬 Discussion Interface")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Introduce New Topic")
        
        # Main topic input - always visible and primary
        topic_input = st.text_area(
            "Enter discussion topic with full context:",
            placeholder="Provide detailed context for meaningful discussion. For example:\n\n'I want to discuss our Q4 budget allocation strategy. We have $2.5M remaining and need to decide between investing in AI infrastructure upgrades versus expanding our marketing team. The AI infrastructure would improve our operational efficiency by an estimated 30%, while the marketing expansion could increase our customer acquisition by 25%. Both initiatives require immediate action to meet our Q1 2026 goals.'",
            height=150,
            key="discussion_topic_input"
        )
        
        # Quick topic suggestions as helper - secondary
        with st.expander("💡 Need inspiration? Click for topic suggestions"):
            st.write("**Quick topic ideas to customize:**")
            quick_topics = [
                "Budget allocation for next quarter - Consider specific amounts and competing priorities",
                "New product development priorities - Include market research findings and timelines", 
                "Market expansion opportunities - Specify target markets and investment requirements",
                "Technology infrastructure upgrades - Detail current limitations and expected ROI",
                "Risk management strategies - Address specific risks and mitigation approaches",
                "Competitive analysis review - Include competitor actions and our response strategy"
            ]
            
            for topic in quick_topics:
                if st.button(topic, key=f"quick_{topic[:20]}", help="Click to use as starting point"):
                    # Set the topic as a starting point that user can edit
                    st.session_state.discussion_topic_input = topic.split(" - ")[0]
                    st.rerun()
        
        if st.button("🎤 Introduce Topic", key="introduce_topic_btn", type="primary"):
            if not topic_input.strip():
                st.error("Please enter a topic with sufficient context for meaningful discussion.")
                return
            
            if len(topic_input.strip()) < 20:
                st.warning("⚠️ Consider adding more context to your topic for richer board discussion.")
            
            try:
                with st.spinner("Introducing topic to board members..."):
                    response = st.session_state.meeting.introduce_topic(topic_input.strip())
                
                st.success("✅ Topic introduced to the board")
                
                # Show immediate feedback about who wants to speak
                if response:
                    st.info(f"📢 {len(response)} board member(s) raised their hand to speak!")
                    for member_name, reason in response:
                        st.write(f"✋ **{member_name}**: {reason}")
                else:
                    st.info("🤔 No board members raised their hands. Try a more engaging topic or ask specific questions.")
                
                st.session_state.discussion_history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "action": "Topic Introduced",
                    "content": topic_input.strip(),
                    "response": response
                })
                
                st.rerun()
            except Exception as e:
                st.error(f"Error introducing topic: {str(e)}")
        
        # Human Chair Controls (only show if human chair)
        if st.session_state.get('is_human_chair', True):
            st.divider()
            st.subheader("👑 Chair Controls")
            
            # Chair comment
            chair_comment = st.text_area(
                "Add your comment as chair:",
                placeholder="Share your perspective, summarize points, or guide the discussion...",
                key="chair_comment_input"
            )
            
            col_chair1, col_chair2 = st.columns(2)
            with col_chair1:
                if st.button("💬 Add Chair Comment", key="chair_comment_btn"):
                    if chair_comment.strip():
                        st.session_state.meeting.chair_comment(chair_comment.strip())
                        st.success("✅ Chair comment added")
                        st.session_state.chair_comment_input = ""  # Clear the input
                        st.rerun()
                    else:
                        st.warning("Please enter a comment")
            
            with col_chair2:
                if st.button("📝 Show Meeting Summary", key="meeting_summary_btn"):
                    participants_summary = st.session_state.meeting.get_participants_summary()
                    st.info(participants_summary)
        
        st.divider()
    
    with col2:
        st.subheader("✋ Speaking Queue")
        
        # Show who wants to speak
        if hasattr(st.session_state.meeting, 'raised_hands') and st.session_state.meeting.raised_hands:
            st.write("**Ready to speak:**")
            
            for member_name, reason in st.session_state.meeting.raised_hands:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"✋ {member_name}: {reason}")
                with col_b:
                    if st.button("🎤", key=f"call_on_{member_name}", help=f"Call on {member_name}"):
                        try:
                            response = st.session_state.meeting.call_on_speaker(member_name)
                            st.success(f"✅ {member_name} has spoken")
                            st.session_state.discussion_history.append({
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "action": "Member Spoke",
                                "speaker": member_name,
                                "response": response
                            })
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error calling on {member_name}: {str(e)}")
        else:
            st.info("No one wants to speak currently.")
        
        # Additional chair actions
        st.subheader("� Chair Tools")
        if st.button("📋 Meeting Minutes", key="minutes_btn"):
            if hasattr(st.session_state.meeting, 'discussion_history'):
                st.info("Meeting minutes functionality coming soon...")
            else:
                st.warning("No discussion history available.")

def expertise_interface():
    """Interface for asking questions to specific experts"""
    if not st.session_state.meeting_started:
        return
    
    st.header("🎯 Expert Consultation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Select expert
        expert_options = {}
        for role_name, member in st.session_state.meeting.participants.items():
            expert_options[f"{member.name} ({member.role.value})"] = member
        
        selected_expert = st.selectbox(
            "Select expert to consult:",
            list(expert_options.keys()),
            key="expert_selector"
        )
        
        # Quick question suggestions based on expert
        if selected_expert:
            expert = expert_options[selected_expert]
            
            quick_questions = {
                ExecutiveRole.CFO: [
                    "What is our current financial position?",
                    "What are the budget implications of this proposal?",
                    "How will this impact our quarterly projections?",
                    "What are the financial risks we should consider?"
                ],
                ExecutiveRole.CTO: [
                    "What are the technical requirements for this initiative?",
                    "How feasible is this from a technology perspective?",
                    "What infrastructure changes would be needed?",
                    "What are the security implications?"
                ],
                ExecutiveRole.CLO: [
                    "Are there any legal risks with this proposal?",
                    "What compliance requirements should we consider?",
                    "Do we need any regulatory approvals?",
                    "What are the liability implications?"
                ],
                ExecutiveRole.CCO: [
                    "How should we communicate this to stakeholders?",
                    "What is the public relations impact?",
                    "How will this affect our brand image?",
                    "What messaging strategy should we use?"
                ],
                ExecutiveRole.CIO: [
                    "What data do we need to make this decision?",
                    "How will this impact our information systems?",
                    "What analytics should we consider?",
                    "How will this affect data security?"
                ]
            }.get(expert.role, ["What is your expert opinion on this matter?"])
            
            selected_quick_question = st.selectbox(
                f"Quick questions for {expert.name}:",
                [""] + quick_questions,
                key="quick_question_selector"
            )
            
            # Handle quick question selection
            if selected_quick_question and not st.session_state.quick_question_selected:
                st.session_state.quick_question_selected = True
                st.rerun()
            
            if st.session_state.quick_question_selected and selected_quick_question:
                question_input = selected_quick_question
            else:
                question_input = st.text_area(
                    f"Ask {expert.name} a question:",
                    placeholder=f"What would you like to ask {expert.name} about their area of expertise?",
                    key="expertise_question_input"
                )
            
            if st.button(f"❓ Ask {expert.name}", key="ask_expert_btn"):
                if not question_input.strip():
                    st.error("Please enter a question.")
                    return
                
                try:
                    response = st.session_state.meeting.ask_specific_question(expert.name, question_input.strip())
                    st.success(f"✅ Question asked to {expert.name}")
                    st.session_state.discussion_history.append({
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "action": "Expert Consultation",
                        "expert": expert.name,
                        "question": question_input.strip(),
                        "response": response
                    })
                    # Reset quick question selection
                    st.session_state.quick_question_selected = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Error asking expert: {str(e)}")
    
    with col2:
        st.subheader("👨‍💼 Expert Profiles")
        
        if selected_expert:
            expert = expert_options[selected_expert]
            role_emoji = {
                ExecutiveRole.CCO: "📢",
                ExecutiveRole.CFO: "💰",
                ExecutiveRole.CTO: "💻",
                ExecutiveRole.CLO: "⚖️",
                ExecutiveRole.CIO: "📊",
                ExecutiveRole.EXECUTIVE_ASSISTANT: "📝"
            }.get(expert.role, "👤")
            
            st.info(f"""
            {role_emoji} **{expert.name}**
            
            **Role:** {expert.role.value}
            
            **Expertise:**
            {', '.join(expert.expertise_areas)}
            
            **Decision Style:**
            {expert.decision_framework.get('style', 'Analytical and data-driven')}
            """)

def voting_interface():
    """Interface for conducting votes on proposals"""
    if not st.session_state.meeting_started:
        return
    
    st.header("🗳️ Voting & Proposals")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Submit Proposal")
        
        # Quick proposal templates
        proposal_templates = [
            "Approve budget allocation of $X for Y initiative",
            "Authorize hiring of Z new employees in Q department", 
            "Proceed with acquisition of Company X",
            "Launch new product line by target date",
            "Implement new technology platform",
            "Restructure organizational reporting lines"
        ]
        
        selected_template = st.selectbox(
            "Proposal templates:",
            [""] + proposal_templates,
            key="proposal_template_selector"
        )
        
        # Handle template selection
        if selected_template and not st.session_state.quick_proposal_selected:
            st.session_state.quick_proposal_selected = True
            st.rerun()
        
        if st.session_state.quick_proposal_selected and selected_template:
            proposal_input = selected_template
        else:
            proposal_input = st.text_area(
                "Enter proposal for voting:",
                placeholder="Describe the proposal you want the board to vote on...",
                key="proposal_input"
            )
        
        if st.button("📤 Submit for Vote", key="submit_proposal_btn"):
            if not proposal_input.strip():
                st.error("Please enter a proposal.")
                return
            
            try:
                results = st.session_state.meeting.call_for_vote(proposal_input.strip())
                st.success("✅ Vote conducted successfully!")
                
                # Store vote results
                vote_record = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "proposal": proposal_input.strip(),
                    "results": results
                }
                st.session_state.votes_history.append(vote_record)
                
                # Display results
                st.subheader("📊 Voting Results")
                
                yes_votes = sum(1 for vote, reasoning in results.values() if vote in ["APPROVE", "SUPPORT"])
                no_votes = sum(1 for vote, reasoning in results.values() if vote == "REJECT")
                conditional_votes = sum(1 for vote, reasoning in results.values() if vote == "CONDITIONAL")
                
                col_yes, col_no, col_conditional = st.columns(3)
                with col_yes:
                    st.metric("✅ YES/APPROVE", yes_votes)
                with col_no:
                    st.metric("❌ REJECT", no_votes)
                with col_conditional:
                    st.metric("⚠️ CONDITIONAL", conditional_votes)
                
                # Detailed results
                for member_name, (vote, reasoning) in results.items():
                    vote_emoji = {"APPROVE": "✅", "REJECT": "❌", "CONDITIONAL": "⚠️", "SUPPORT": "👍"}.get(vote, "🗳️")
                    with st.expander(f"{vote_emoji} {vote} - {member_name}"):
                        st.write(f"**Reasoning:** {reasoning}")
                
                # Human Chair Voting (if applicable)
                if st.session_state.get('is_human_chair', True):
                    st.divider()
                    st.subheader("👑 Chair Vote")
                    
                    chair_vote_col1, chair_vote_col2 = st.columns([1, 2])
                    with chair_vote_col1:
                        chair_vote = st.selectbox(
                            "Your vote:",
                            ["APPROVE", "REJECT", "CONDITIONAL", "SUPPORT"],
                            key="chair_vote_selection"
                        )
                    
                    with chair_vote_col2:
                        chair_reasoning = st.text_area(
                            "Your reasoning:",
                            placeholder="Explain your vote as chair...",
                            key="chair_vote_reasoning"
                        )
                    
                    if st.button("🗳️ Cast Chair Vote", key="cast_chair_vote_btn"):
                        if chair_reasoning.strip():
                            # Add chair vote to results
                            results[st.session_state.meeting.chair.name] = (chair_vote, chair_reasoning.strip())
                            
                            # Display chair vote
                            chair_vote_emoji = {"APPROVE": "✅", "REJECT": "❌", "CONDITIONAL": "⚠️", "SUPPORT": "👍"}.get(chair_vote, "🗳️")
                            st.success(f"👑 Chair Vote Recorded: {chair_vote_emoji} {chair_vote}")
                            st.info(f"**Chair Reasoning:** {chair_reasoning.strip()}")
                            
                            # Update vote record with chair vote
                            vote_record["results"] = results
                            st.session_state.votes_history[-1] = vote_record
                            
                            # Add to chat log
                            st.session_state.meeting.chair_comment(f"My vote: {chair_vote} - {chair_reasoning.strip()}")
                            
                            st.rerun()
                        else:
                            st.warning("Please provide reasoning for your vote")
                
                # Reset template selection
                st.session_state.quick_proposal_selected = False
                st.rerun()
                
            except Exception as e:
                st.error(f"Error conducting vote: {str(e)}")
    
    with col2:
        st.subheader("📋 Proposal Guidelines")
        st.info("""
        **Good proposals are:**
        - Clear and specific
        - Actionable
        - Include relevant details
        - Address key concerns
        
        **Example:**
        "Approve $500K budget for Q4 marketing campaign to increase brand awareness by 25% through digital advertising and influencer partnerships."
        """)
        
        if selected_template:
            st.subheader("📝 Template Selected")
            with st.expander("Customize this template"):
                st.write(f"**Template:** {selected_template}")
                st.write("Copy this template to the proposal box above and customize it.")

def display_live_chat():
    """Display the live chat from the meeting"""
    if not st.session_state.meeting_started:
        return
    
    st.subheader("💬 Live Meeting Chat")
    
    # Check if meeting exists and has chat log
    if not st.session_state.meeting:
        st.info("No active meeting")
        return
    
    if not hasattr(st.session_state.meeting, 'chat_log'):
        st.info("No chat messages yet")
        return
    
    # Create a container for the chat
    chat_container = st.container()
    
    with chat_container:
        # Display recent chat messages (last 10)
        recent_messages = st.session_state.meeting.chat_log[-10:] if st.session_state.meeting.chat_log else []
        
        if not recent_messages:
            st.info("No chat messages yet. Start a discussion or ask questions!")
            return
        
        for message in recent_messages:
            timestamp = message.get("timestamp", "")
            speaker = message.get("speaker", "Unknown")
            content = message.get("message", "")
            msg_type = message.get("type", "speech")
            
            if msg_type == "system":
                st.info(f"🔔 [{timestamp}] **SYSTEM**: {content}")
            elif msg_type == "chair":
                st.success(f"👑 [{timestamp}] **{speaker}**: {content}")
            else:
                # Find role emoji
                role_emoji = "👤"
                try:
                    if hasattr(st.session_state.meeting, 'participants'):
                        for member in st.session_state.meeting.participants.values():
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
                except:
                    pass  # If there's any error, just use default emoji
                    
                st.write(f"{role_emoji} [{timestamp}] **{speaker}**: {content}")
        
        # Add some space at the bottom
        st.write("")
    
    # Export chat button
    if st.session_state.meeting.chat_log:
        if st.button("📥 Export Chat Log", key="export_chat_btn"):
            try:
                chat_export = st.session_state.meeting.export_chat_log()
                st.download_button(
                    label="💾 Download Chat Log",
                    data=chat_export,
                    file_name=f"board_meeting_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_chat_btn"
                )
            except Exception as e:
                st.error(f"Error exporting chat: {str(e)}")

def display_voting_history():
    """Display the voting history"""
    if not st.session_state.votes_history:
        return
    
    st.subheader("🗳️ Voting History")
    
    for i, vote_record in enumerate(reversed(st.session_state.votes_history)):
        with st.expander(f"Vote #{len(st.session_state.votes_history) - i} - {vote_record['timestamp']}"):
            st.write(f"**Proposal:** {vote_record['proposal']}")
            
            results = vote_record['results']
            yes_votes = sum(1 for vote, reasoning in results.values() if vote in ["APPROVE", "SUPPORT"])
            no_votes = sum(1 for vote, reasoning in results.values() if vote == "REJECT")
            conditional_votes = sum(1 for vote, reasoning in results.values() if vote == "CONDITIONAL")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("✅ YES/APPROVE", yes_votes)
            with col2:
                st.metric("❌ REJECT", no_votes)
            with col3:
                st.metric("⚠️ CONDITIONAL", conditional_votes)
            
            for member_name, (vote, reasoning) in results.items():
                vote_emoji = {"APPROVE": "✅", "REJECT": "❌", "CONDITIONAL": "⚠️", "SUPPORT": "👍"}.get(vote, "🗳️")
                st.write(f"**{member_name}:** {vote_emoji} {vote} - {reasoning}")

def display_discussion_history():
    """Display the discussion history"""
    if not st.session_state.discussion_history:
        return
    
    st.subheader("📝 Discussion History")
    
    for i, discussion in enumerate(reversed(st.session_state.discussion_history)):
        with st.expander(f"{discussion['action']} - {discussion['timestamp']}"):
            
            if discussion['action'] == "Topic Introduced":
                st.write(f"**Topic:** {discussion['content']}")
                if 'response' in discussion:
                    st.write(f"**Response:** {discussion['response']}")
            
            elif discussion['action'] == "Member Spoke":
                st.write(f"**Speaker:** {discussion['speaker']}")
                if 'response' in discussion:
                    st.write(f"**Response:** {discussion['response']}")
            
            elif discussion['action'] == "Expert Consultation":
                st.write(f"**Expert:** {discussion['expert']}")
                st.write(f"**Question:** {discussion['question']}")
                if 'response' in discussion:
                    st.write(f"**Response:** {discussion['response']}")
            
            elif discussion['action'] == "Chair Comment":
                st.write(f"**Comment:** {discussion['content']}")

def main():
    """Main Streamlit application"""
    init_session_state()
    
    # Main title
    st.title("🏢 AI Board Meeting System")
    st.markdown("---")
    
    # Sidebar for meeting status
    display_meeting_status()
    
    # Main content area
    if not st.session_state.meeting_started:
        create_meeting_interface()
    else:
        # Tabs for different interfaces
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Discussion", "🎯 Expert Consultation", "🗳️ Voting", "📊 History"])
        
        with tab1:
            discussion_interface()
            st.markdown("---")
            display_live_chat()
        
        with tab2:
            expertise_interface()
        
        with tab3:
            voting_interface()
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                display_discussion_history()
            with col2:
                display_voting_history()
        
        # Reset meeting button in sidebar
        st.sidebar.markdown("---")
        if st.sidebar.button("🔄 End Meeting", key="end_meeting_btn"):
            st.session_state.meeting_started = False
            st.session_state.meeting = None
            st.session_state.discussion_history = []
            st.session_state.votes_history = []
            st.rerun()

if __name__ == "__main__":
    main()
