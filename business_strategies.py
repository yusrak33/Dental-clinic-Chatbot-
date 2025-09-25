"""
Business Strategies Module for Dental Clinic Chatbot
Handles all business strategy related queries and calculations
"""

import logging

logger = logging.getLogger(__name__)

def handle_business_strategy_query(user_message, df, doctor_names):
    """
    Handle business strategy related queries and return appropriate strategies
    
    Args:
        user_message (str): The user's query
        df (pd.DataFrame): The clinic dataset
        doctor_names (list): List of available doctor names
    
    Returns:
        str: Business strategy response
    """
    msg_lower = user_message.lower()
    
    # Collect clinic statistics
    stats = {
        'unique_patients': df['patient_name'].nunique() if 'patient_name' in df.columns else 0,
        'unique_doctors': len(doctor_names),
        'doctor_names': doctor_names,
        'unique_invoices': df['invoice_id'].nunique() if 'invoice_id' in df.columns else 0,
        'total_price': df['price'].sum() if 'price' in df.columns else 0
    }

    # Business calculation constants
    avg_fee = 5000
    recall_rate = 0.3
    membership_fee = 20000
    referral_avg_fee = 15000
    
    # Calculate revenue projections
    recall_patients = int(stats['unique_patients'] * recall_rate)
    recall_revenue = recall_patients * avg_fee
    membership_revenue = 100 * membership_fee  # assume 100 signups
    referral_revenue = 50 * referral_avg_fee   # assume 50 referrals
    retained_patients = int(stats['unique_patients'] * 0.1)
    retention_revenue = retained_patients * avg_fee

    # Determine strategy focus based on user query
    if "revenue" in msg_lower or "income" in msg_lower:
        return get_revenue_strategy(recall_patients, recall_revenue, membership_fee, membership_revenue, 
                                  referral_avg_fee, referral_revenue)
    
    elif "retention" in msg_lower or "loyalty" in msg_lower:
        return get_retention_strategy(retained_patients, retention_revenue)
    
    elif "marketing" in msg_lower or "advertising" in msg_lower:
        return get_marketing_strategy()
    
    else:
        return get_comprehensive_strategy(recall_patients, recall_revenue, membership_revenue, 
                                        referral_revenue)

def get_revenue_strategy(recall_patients, recall_revenue, membership_fee, membership_revenue, 
                        referral_avg_fee, referral_revenue):
    """Generate revenue-focused strategy"""
    return f"""
📈 Data-Driven Revenue Strategies for Your Clinic:

1. Proactive Recall System
   • Target: {recall_patients} patients (30% of current base)
   • Potential Revenue: ≈ Rs. {recall_revenue:,}

2. Membership Plans
   • Target: 100 patients at Rs. {membership_fee:,}/year
   • Recurring Revenue: ≈ Rs. {membership_revenue:,}

3. Referral Program
   • Target: 50 referrals at Rs. {referral_avg_fee:,} average
   • New Revenue: ≈ Rs. {referral_revenue:,}

💡 Total Potential: Rs. {recall_revenue + membership_revenue + referral_revenue:,} additional revenue
"""

def get_retention_strategy(retained_patients, retention_revenue):
    """Generate patient retention strategy"""
    return f"""
🤝 Patient Retention Strategies:

Revenue Impact:
• 10% retention improvement = {retained_patients} patients
• Additional yearly revenue: ≈ Rs. {retention_revenue:,}

Action Plan:
✅ Personalized Reminders - SMS/Email follow-ups
✅ Post-Treatment Follow-ups - Care & satisfaction calls
✅ Rewards & Loyalty Program - Points for visits
✅ Patient Education Content - Oral health tips
✅ Feedback Response System - Address concerns quickly

Pro Tip: Focus on patients with high-value treatments for maximum retention ROI.
"""

def get_marketing_strategy():
    """Generate marketing strategy"""
    return f"""
📢 Marketing Strategies for Clinic Growth:

Digital Presence:
• Optimize Google Maps listing + collect reviews
• Run targeted Facebook/Instagram ads for local area
• Create engaging content about dental health

Service Promotion:
• Showcase cosmetic treatments (whitening, aligners, veneers)
• Highlight preventive care benefits
• Promote family dental packages

Community Outreach:
• Organize free dental check-up camps
• Partner with nearby schools for children's dental programs
• Collaborate with local companies for employee health programs

Budget Allocation: 70% digital, 20% community events, 10% traditional advertising
"""

def get_comprehensive_strategy(recall_patients, recall_revenue, membership_revenue, referral_revenue):
    """Generate comprehensive growth strategy"""
    return f"""
🚀 Comprehensive Growth Strategies:

Multi-Channel Approach:
• Patient Recall: Target {recall_patients} patients for Rs. {recall_revenue:,} revenue
• Membership Plans: 100 members for Rs. {membership_revenue:,} recurring income
• Referral Program: 50 referrals for Rs. {referral_revenue:,} new business

Service Expansion:
• High-value services: Implants, veneers, aligners
• Cosmetic dentistry: Teeth whitening, smile makeovers
• Preventive care: Regular cleanings, fluoride treatments

Business Development:
• Improve online reputation & SEO ranking
• Offer bundled family care plans
• Implement patient education programs
• Develop corporate tie-ups for employee benefits

Expected ROI: 3-5x investment within 12 months with proper execution.
"""

def is_business_strategy_query(user_message):
    """
    Check if the user message is related to business strategies
    
    Args:
        user_message (str): The user's query
    
    Returns:
        bool: True if it's a business strategy query, False otherwise
    """
    strategy_keywords = ["strategy", "growth", "increase patients", "marketing", "revenue", "retention"]
    msg_lower = user_message.lower()
    return any(keyword in msg_lower for keyword in strategy_keywords)
