# migraine_recommendation_engine.py
"""
Recommendation engine for migraine prevention and management
Generates personalized, actionable recommendations based on predictions
"""

class MigraineRecommendationEngine:
    """
    Generates personalized, actionable recommendations based on migraine predictions
    and user symptoms
    """
    
    def __init__(self):
        self.recommendation_templates = {
            'immediate_relief': {
                'power_nap': {
                    'trigger_conditions': ['fatigue', 'sleep_deprivation', 'high_stress'],
                    'action': "Take a 20-minute power nap in a dark, quiet room",
                    'rationale': "Short rest can help prevent migraine escalation",
                    'priority': 'high'
                },
                'hydration': {
                    'trigger_conditions': ['dehydration', 'all'],
                    'action': "Drink 2 glasses of water immediately",
                    'rationale': "Dehydration is a common migraine trigger",
                    'priority': 'high'
                },
                'caffeine': {
                    'trigger_conditions': ['withdrawal', 'low_energy'],
                    'action': "Have a small cup of coffee or tea",
                    'rationale': "Small amount of caffeine can provide relief",
                    'priority': 'medium'
                }
            },
            
            'preventive_actions': {
                'walk': {
                    'trigger_conditions': ['sedentary', 'stress', 'tension'],
                    'action': "Take a 10-minute gentle walk outside",
                    'rationale': "Light activity reduces stress and improves circulation",
                    'priority': 'medium'
                },
                'snack': {
                    'trigger_conditions': ['hunger', 'low_blood_sugar', 'missed_meals'],
                    'action': "Eat a balanced snack with protein and complex carbs",
                    'rationale': "Stabilizes blood sugar and prevents hunger triggers",
                    'priority': 'high'
                },
                'breathing': {
                    'trigger_conditions': ['stress', 'anxiety', 'tension'],
                    'action': "Practice 5 minutes of deep breathing exercises",
                    'rationale': "Reduces stress and muscle tension",
                    'priority': 'medium'
                }
            },
            
            'environmental_adjustments': {
                'light_sensitivity': {
                    'trigger_conditions': ['light_sensitivity', 'visual_aura'],
                    'action': "Move to a dimly lit room, use blue light filters",
                    'rationale': "Reduces photophobia and visual triggers",
                    'priority': 'high'
                },
                'noise_reduction': {
                    'trigger_conditions': ['sound_sensitivity', 'concentration'],
                    'action': "Use noise-cancelling headphones or earplugs",
                    'rationale': "Minimizes auditory triggers",
                    'priority': 'medium'
                }
            },
            
            'medication_readiness': {
                'early_medication': {
                    'trigger_conditions': ['high_risk', 'early_symptoms'],
                    'action': "Take prescribed medication at first signs",
                    'rationale': "Early intervention increases effectiveness",
                    'priority': 'high'
                },
                'rescue_kit': {
                    'trigger_conditions': ['all'],
                    'action': "Keep migraine rescue kit accessible",
                    'rationale': "Preparation reduces stress during onset",
                    'priority': 'low'
                }
            }
        }
        
        self.risk_level_actions = {
            'green': {
                'focus': 'Maintenance & Prevention',
                'primary_actions': ['hydration', 'regular_meals', 'sleep_schedule'],
                'intensity': 'normal'
            },
            'yellow': {
                'focus': 'Early Intervention',
                'primary_actions': ['snack', 'hydration', 'breathing', 'walk'],
                'intensity': 'moderate'
            },
            'orange': {
                'focus': 'Active Prevention',
                'primary_actions': ['power_nap', 'medication_readiness', 'environment_adjustment'],
                'intensity': 'high'
            },
            'red': {
                'focus': 'Immediate Action',
                'primary_actions': ['medication', 'dark_room', 'rest', 'hydration'],
                'intensity': 'urgent'
            }
        }

    def analyze_user_context(self, features, symptoms=None, triggers=None):
        """
        Analyze user context to personalize recommendations
        """
        context = {
            'high_stress': features.get('stress_level', 0) > 6,
            'sleep_deprivation': features.get('total_meals', 3) < 2,  # Proxy for sleep issues
            'missed_meals': features.get('meals_skipped', 0) > 0,
            'sedentary': features.get('activity_index', 0) < 30,
            'dehydration': True,  # Always recommend hydration
            'fatigue': features.get('mood_value', 5) < 4  # Low mood as proxy for fatigue
        }
        
        # Add symptom-based context
        if symptoms:
            symptom_list = [s.strip().lower() for s in str(symptoms).split(',')]
            context.update({
                'light_sensitivity': any(s in ['light sensitivity', 'photophobia'] for s in symptom_list),
                'sound_sensitivity': any(s in ['sound sensitivity', 'phonophobia'] for s in symptom_list),
                'visual_aura': any(s in ['aura', 'visual disturbances'] for s in symptom_list),
                'nausea': 'nausea' in symptom_list
            })
        
        return context

    def generate_personalized_recommendations(self, prediction_output, features, user_symptoms=None, user_triggers=None):
        """
        Generate concise, personalized recommendations based on model predictions
        """
        risk_band = prediction_output['risk_band']
        probability = prediction_output['probability']
        intensity = prediction_output['predicted_intensity']
        
        # Analyze user context
        context = self.analyze_user_context(features, user_symptoms, user_triggers)
        
        # Get risk-appropriate actions
        risk_actions = self.risk_level_actions[risk_band]
        
        recommendations = []
        
        # Immediate relief actions for medium-high risk
        if risk_band in ['orange', 'red']:
            immediate_recs = self._get_immediate_actions(context, intensity)
            recommendations.extend(immediate_recs)
        
        # Preventive actions for all risk levels
        preventive_recs = self._get_preventive_actions(context, risk_band)
        recommendations.extend(preventive_recs)
        
        # Environmental adjustments based on symptoms
        if user_symptoms:
            environmental_recs = self._get_environmental_actions(context)
            recommendations.extend(environmental_recs)
        
        # Medication readiness for high risk
        if risk_band in ['orange', 'red'] and intensity > 5:
            med_recs = self._get_medication_actions(context)
            recommendations.extend(med_recs)
        
        # Limit to top 3-5 most relevant recommendations
        recommendations = self._prioritize_recommendations(recommendations, risk_band)
        
        return {
            'risk_level': risk_band,
            'focus_area': risk_actions['focus'],
            'key_recommendations': recommendations[:4],  # Top 4 most important
            'quick_actions': self._generate_quick_actions(risk_band),
            'prevention_tips': self._generate_prevention_tips(context)
        }

    def _get_immediate_actions(self, context, intensity):
        """Get immediate relief actions based on context"""
        actions = []
        
        if context.get('high_stress') or context.get('fatigue'):
            actions.append({
                'type': 'power_nap',
                'action': "Rest for 20 minutes in a quiet, dark space",
                'reason': "Reduces sensory overload and stress",
                'urgency': 'high' if intensity > 5 else 'medium'
            })
        
        if context.get('dehydration'):
            actions.append({
                'type': 'hydration',
                'action': "Drink water with electrolyte powder",
                'reason': "Prevents dehydration-related triggers",
                'urgency': 'high'
            })
        
        if context.get('missed_meals') or context.get('fatigue'):
            actions.append({
                'type': 'snack',
                'action': "Eat a banana with almond butter or similar balanced snack",
                'reason': "Stabilizes blood sugar levels",
                'urgency': 'medium'
            })
        
        return actions

    def _get_preventive_actions(self, context, risk_band):
        """Get preventive actions based on risk level and context"""
        actions = []
        
        if context.get('sedentary') and risk_band in ['yellow', 'orange']:
            actions.append({
                'type': 'movement',
                'action': "Take a 10-minute gentle walk",
                'reason': "Improves circulation and reduces tension",
                'urgency': 'medium'
            })
        
        if context.get('high_stress'):
            actions.append({
                'type': 'stress_relief',
                'action': "Practice 4-7-8 breathing (inhale 4s, hold 7s, exhale 8s)",
                'reason': "Activates relaxation response",
                'urgency': 'medium'
            })
        
        # Always include basic prevention
        actions.append({
            'type': 'hydration_maintenance',
            'action': "Sip water regularly throughout day",
            'reason': "Maintains consistent hydration",
            'urgency': 'low'
        })
        
        return actions

    def _get_environmental_actions(self, context):
        """Get environmental adjustment recommendations"""
        actions = []
        
        if context.get('light_sensitivity'):
            actions.append({
                'type': 'light_management',
                'action': "Wear sunglasses or use blue light filter",
                'reason': "Reduces photophobia triggers",
                'urgency': 'high'
            })
        
        if context.get('sound_sensitivity'):
            actions.append({
                'type': 'sound_management',
                'action': "Use noise-cancelling headphones in noisy environments",
                'reason': "Minimizes auditory stimulation",
                'urgency': 'medium'
            })
        
        return actions

    def _get_medication_actions(self, context):
        """Get medication-related recommendations"""
        actions = []
        
        actions.append({
            'type': 'medication_readiness',
            'action': "Keep abortive medication easily accessible",
            'reason': "Allows quick intervention if migraine develops",
            'urgency': 'high'
        })
        
        return actions

    def _prioritize_recommendations(self, recommendations, risk_band):
        """Prioritize recommendations based on urgency and risk level"""
        urgency_order = {'high': 3, 'medium': 2, 'low': 1}
        
        # Sort by urgency (high first), then by type
        recommendations.sort(key=lambda x: (urgency_order[x['urgency']], x['type']), reverse=True)
        
        return recommendations

    def _generate_quick_actions(self, risk_band):
        """Generate simple, immediate actions user can take right now"""
        quick_actions = {
            'green': [
                "Drink a glass of water",
                "Take 5 deep breaths",
                "Check your posture"
            ],
            'yellow': [
                "Have a healthy snack",
                "Step outside for fresh air",
                "Do 2 minutes of neck stretches"
            ],
            'orange': [
                "Rest in quiet space for 15 minutes",
                "Take prescribed preventive medication",
                "Apply cold compress to neck"
            ],
            'red': [
                "Take abortive medication immediately",
                "Lie down in dark room",
                "Use emergency migraine protocol"
            ]
        }
        
        return quick_actions.get(risk_band, quick_actions['yellow'])

    def _generate_prevention_tips(self, context):
        """Generate personalized prevention tips based on user context"""
        tips = []
        
        if context.get('missed_meals'):
            tips.append("🍽️ Eat small, frequent meals to maintain blood sugar")
        
        if context.get('high_stress'):
            tips.append("😌 Schedule 5-minute stress breaks every 2 hours")
        
        if context.get('sedentary'):
            tips.append("🚶‍♂️ Set reminder to stand and stretch every hour")
        
        if context.get('sleep_deprivation'):
            tips.append("💤 Aim for consistent sleep schedule")
        
        # Always include these basics
        tips.extend([
            "💧 Keep water bottle visible as hydration reminder",
            "☀️ Wear sunglasses in bright light if light-sensitive"
        ])
        
        return tips[:3]  # Return top 3 most relevant


def format_recommendations_for_display(prediction_output):
    """
    Format recommendations for user-friendly display
    """
    output = []
    
    # Risk level header
    risk_colors = {
        'green': '🟢', 'yellow': '🟡', 'orange': '🟠', 'red': '🔴'
    }
    
    output.append(f"{risk_colors[prediction_output['risk_band']]} "
                 f"{prediction_output['risk_label']} - {prediction_output['probability']}% probability")
    
    # Key recommendations
    output.append("\n🎯 **Key Actions:**")
    for i, rec in enumerate(prediction_output['key_recommendations'], 1):
        output.append(f"{i}. {rec['action']}")
        output.append(f"   💡 Why: {rec['reason']}")
    
    # Quick actions
    output.append("\n⚡ **Quick Steps Now:**")
    for action in prediction_output['quick_actions']:
        output.append(f"• {action}")
    
    # Prevention tips
    output.append("\n🛡️ **Prevention Tips:**")
    for tip in prediction_output['prevention_tips']:
        output.append(f"• {tip}")
    
    # Intensity and duration info
    if prediction_output['predicted_intensity'] > 0:
        output.append(f"\n📊 Expected intensity: {prediction_output['intensity_label']} "
                     f"({prediction_output['predicted_intensity']}/10)")
        output.append(f"⏱️ Expected duration: {prediction_output['predicted_duration_minutes']} minutes")
    
    return "\n".join(output)
