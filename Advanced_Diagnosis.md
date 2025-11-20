# Comprehensive Trigger Categories
![Comprehensive Trigger Categories](/Diagrams/Comprehensive_Triger_Categories.png)

# Enhanced Data Collection System

1. Comprehensive User Onboarding Questionnaire
```
@Data
public class ComprehensiveUserProfile {
    // Basic Information
    private String id;
    private Gender gender;
    private LocalDate birthDate;
    private Double height;
    private Double weight;
    
    // Medical History
    private List<String> comorbidities; // IBS, depression, anxiety, etc.
    private List<String> allergies;
    private String familyHistory; // Migraine in family?
    private Integer yearsWithMigraines;
    
    // Migraine Characteristics
    private MigraineType migraineType; // WITH_AURA, WITHOUT_AURA, CHRONIC, etc.
    private List<String> typicalSymptoms;
    private String usualDuration; // hours
    private String painLocation; // unilateral, bilateral, etc.
    
    // Confirmed Triggers (User-Identified)
    private Map<String, TriggerSeverity> confirmedTriggers;
    
    // Lifestyle Patterns
    private SleepPattern sleepPattern;
    private ExerciseFrequency exerciseFrequency;
    private WorkSchedule workSchedule;
    private StressManagement stressManagement;
}
```

2. Enhanced Daily Check-in Form

```
@Data
public class ComprehensiveDailyCheckin {
    private String userId;
    private LocalDateTime timestamp;
    
    // Physical State
    private Integer sleepQuality; // 1-10
    private Integer sleepDuration; // hours
    private Boolean napTaken;
    private Integer hydrationLevel; // glasses of water
    private String mealRegularity; // regular, skipped, irregular
    
    // Environmental Exposure
    private Integer screenTime; // hours
    private Boolean brightLightExposure;
    private Boolean loudNoiseExposure;
    private Boolean strongSmellExposure;
    private String weatherSensitivity; // none, mild, moderate, severe
    
    // Dietary Tracking
    private List<String> consumedFoods;
    private List<String> consumedBeverages;
    private Boolean artificialSweeteners;
    private Boolean processedFoods;
    private Boolean caffeineConsumed;
    private Integer alcoholUnits;
    
    // Physical Activity
    private Integer exerciseDuration; // minutes
    private String exerciseIntensity; // light, moderate, intense
    private Boolean neckStrain; // from posture, etc.
    
    // Psychological State
    private Integer stressLevel; // 1-10
    private String mood; // happy, anxious, depressed, calm
    private String emotionalTriggers; // work, family, social, etc.
    
    // Hormonal Tracking (for women)
    private Integer menstrualCycleDay;
    private String cyclePhase; // follicular, ovulation, luteal, menstrual
    private Boolean hormonalMedication;
    
    // Early Warning Signs
    private List<String> prodromeSymptoms; // yawning, neck pain, mood changes
    private List<String> auraSymptoms; // visual, sensory, speech
    
    // Current Status
    private Boolean migraineOccurred;
    private Integer migraineIntensity; // 1-10
    private List<String> migraineSymptoms;
    private String medicationsTaken;
}

```

# Personalized Trigger Scoring System

```
@Service
public class AdvancedFeatureCalculator {
    
    public Map<String, Double> calculatePersonalizedFeatures(UserProfile user, 
                                                           DailyCheckin checkin,
                                                           WeatherData weather) {
        Map<String, Double> features = new HashMap<>();
        
        // 1. Environmental Triggers (Enhanced)
        features.put("pressure_change_24h", calculatePressureChange(weather));
        features.put("temperature_variation_12h", calculateTempVariation(weather));
        features.put("humidity_risk", calculateHumidityRisk(weather));
        features.put("air_quality_index", getAirQualityIndex(weather));
        
        // 2. Dietary Trigger Matrix
        features.put("dietary_risk_score", calculateDietaryRisk(
            checkin.getConsumedFoods(), 
            checkin.getConsumedBeverages(),
            user.getFoodSensitivities()
        ));
        
        features.put("caffeine_withdrawal_risk", calculateCaffeineRisk(
            checkin.getCaffeineConsumed(),
            user.getTypicalCaffeineIntake()
        ));
        
        // 3. Sleep Pattern Analysis
        features.put("sleep_debt_score", calculateSleepDebt(
            checkin.getSleepDuration(),
            user.getOptimalSleepDuration()
        ));
        
        features.put("sleep_consistency_score", calculateSleepConsistency(
            checkin.getSleepQuality(),
            user.getHistoricalSleepPatterns()
        ));
        
        // 4. Stress & Psychological Factors
        features.put("cumulative_stress_score", calculateCumulativeStress(
            checkin.getStressLevel(),
            checkin.getMood(),
            user.getStressThreshold()
        ));
        
        features.put("emotional_trigger_score", calculateEmotionalTriggers(
            checkin.getEmotionalTriggers(),
            user.getEmotionalSensitivities()
        ));
        
        // 5. Physical Factors
        features.put("posture_strain_score", calculatePostureRisk(
            checkin.getNeckStrain(),
            checkin.getScreenTime()
        ));
        
        features.put("exercise_balance_score", calculateExerciseBalance(
            checkin.getExerciseDuration(),
            checkin.getExerciseIntensity(),
            user.getExerciseTolerance()
        ));
        
        // 6. Hormonal Factors (Gender-Specific)
        if (user.getGender() == Gender.FEMALE) {
            features.put("hormonal_risk_factor", calculateHormonalRisk(
                checkin.getMenstrualCycleDay(),
                checkin.getCyclePhase(),
                user.getHormonalPatterns()
            ));
        }
        
        // 7. Sensory Sensitivity Scores
        features.put("sensory_overload_score", calculateSensoryOverload(
            checkin.getBrightLightExposure(),
            checkin.getLoudNoiseExposure(),
            checkin.getStrongSmellExposure(),
            user.getSensorySensitivities()
        ));
        
        // 8. Personal Trigger Patterns
        features.put("personal_trigger_alignment", calculatePersonalTriggerAlignment(
            features,
            user.getConfirmedTriggers()
        ));
        
        return features;
    }
}

```

# Trigger Pattern Recognition

```
@Service
public class PersonalTriggerLearner {
    
    public void learnFromUserPatterns(String userId) {
        List<DailyCheckin> userHistory = checkinRepository.findByUserId(userId);
        List<MigraineOccurrence> migraines = migraineRepository.findByUserId(userId);
        
        // Analyze patterns before migraines
        Map<String, Double> triggerCorrelations = new HashMap<>();
        
        for (MigraineOccurrence migraine : migraines) {
            // Look at 24-48 hours before migraine
            LocalDateTime windowStart = migraine.getStartTime().minusHours(48);
            LocalDateTime windowEnd = migraine.getStartTime().minusHours(1);
            
            List<DailyCheckin> preMigraineData = filterCheckinsInWindow(
                userHistory, windowStart, windowEnd);
            
            // Analyze what was different before migraine
            analyzePreMigrainePatterns(preMigraineData, triggerCorrelations);
        }
        
        // Compare with non-migraine days
        analyzeNonMigrainePatterns(userHistory, migraines, triggerCorrelations);
        
        // Update user's personal trigger profile
        updatePersonalTriggerProfile(userId, triggerCorrelations);
    }
    
    private void analyzePreMigrainePatterns(List<DailyCheckin> preMigraineData, 
                                          Map<String, Double> correlations) {
        for (DailyCheckin checkin : preMigraineData) {
            // Analyze each potential trigger
            analyzeSleepPatterns(checkin, correlations);
            analyzeDietaryPatterns(checkin, correlations);
            analyzeStressPatterns(checkin, correlations);
            analyzeEnvironmentalFactors(checkin, correlations);
            analyzePhysicalFactors(checkin, correlations);
        }
    }
}

```

# Multi-Factor Risk Assessment

```
@Service
public class AdvancedPredictionService {
    
    public ComprehensiveRiskAssessment predictMigraineRisk(UserProfile user, 
                                                         DailyCheckin checkin) {
        // Calculate all feature scores
        Map<String, Double> features = featureCalculator.calculatePersonalizedFeatures(
            user, checkin, weatherService.getCurrentWeather());
        
        // Get base ML prediction
        Double baseProbability = mlService.predict(features);
        
        // Apply personalization factors
        Double personalizedProbability = applyPersonalization(
            baseProbability, user, features);
        
        // Identify top contributing factors
        List<TriggerContribution> topContributors = identifyTopContributors(
            features, user.getPersonalTriggerWeights());
        
        // Generate personalized recommendations
        List<String> recommendations = generatePersonalizedRecommendations(
            personalizedProbability, topContributors, user);
        
        return ComprehensiveRiskAssessment.builder()
            .probability(personalizedProbability)
            .riskLevel(calculateRiskLevel(personalizedProbability))
            .topContributors(topContributors)
            .recommendations(recommendations)
            .confidenceScore(calculateConfidence(features, user))
            .timeframe(estimateTimeframe(topContributors))
            .build();
    }
    
    private Double applyPersonalization(Double baseProbability, 
                                      UserProfile user, 
                                      Map<String, Double> features) {
        Double personalized = baseProbability;
        
        // Adjust based on user's historical accuracy with similar patterns
        Double historicalAccuracy = user.getPatternAccuracy().getOrDefault(
            getCurrentPatternSignature(features), 1.0);
        
        // Adjust based on trigger severity for this user
        for (Map.Entry<String, Double> entry : features.entrySet()) {
            String trigger = entry.getKey();
            Double severity = user.getPersonalTriggerWeights().getOrDefault(trigger, 1.0);
            personalized *= (1 + (severity * entry.getValue()));
        }
        
        return Math.min(personalized, 0.95); // Cap at 95%
    }
}

```

# Comprehensive Tracking Dashboard

```
@Data
public class PersonalInsightsDashboard {
    private RiskAssessment currentRisk;
    private List<TriggerInsight> personalInsights;
    private Map<String, Double> triggerSensitivities;
    private List<PatternDiscovery> discoveredPatterns;
    private ProgressMetrics improvementMetrics;
}

@Data
public class TriggerInsight {
    private String triggerCategory;
    private Double personalSensitivity;
    private Double populationAverage;
    private String insightDescription;
    private List<String> managementTips;
}

@Data
public class PatternDiscovery {
    private String patternDescription;
    private Double confidence;
    private Integer occurrences;
    private String timePattern; // e.g., "Usually occurs 24h after poor sleep"
    private List<String> contributingFactors;
}

```

# Feedback Loop for Improvement 

```
@Service
public class ContinuousLearningService {
    
    @Scheduled(fixedRate = 86400000) // Daily
    public void updatePersonalModels() {
        List<UserProfile> activeUsers = userRepository.findActiveUsers();
        
        for (UserProfile user : activeUsers) {
            if (hasSufficientNewData(user)) {
                // Retrain personal trigger weights
                updatePersonalTriggerWeights(user);
                
                // Discover new patterns
                discoverNewPatterns(user);
                
                // Validate prediction accuracy
                calculatePredictionAccuracy(user);
                
                // Update ML model if significant improvements
                if (hasSignificantNewPatterns(user)) {
                    retrainPersonalModel(user);
                }
            }
        }
    }
    
    private void discoverNewPatterns(UserProfile user) {
        // Use association rule mining to find new trigger combinations
        List<DailyCheckin> recentData = getRecentUserData(user, 90); // 90 days
        
        Map<String, Double> newPatterns = patternMiner.discoverAssociations(
            recentData, user.getMigraineOccurrences());
        
        // Only keep patterns with high confidence
        newPatterns.entrySet().removeIf(entry -> entry.getValue() < 0.7);
        
        // Update user profile with new discoveries
        user.getDiscoveredPatterns().putAll(newPatterns);
        userRepository.save(user);
    }
}

```

# Personal Migraine Profile

```
@Data
public class PersonalMigraineProfile {
    private String userId;
    private Map<String, Double> triggerSensitivities;
    private List<MigrainePattern> commonPatterns;
    private Map<String, Double> predictionAccuracy;
    private List<ImprovementMetric> progressMetrics;
    private PersonalBaseline baselineMetrics;
}

@Data
public class MigrainePattern {
    private String patternId;
    private String description;
    private Double confidence;
    private Integer frequency;
    private List<String> triggerCombination;
    private String typicalTiming;
    private String severityPattern;
}

```
