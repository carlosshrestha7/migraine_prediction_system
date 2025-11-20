# Project Name: personalized migraine prediction system

## Vision: 
A personalized migraine prediction system using Java/Spring Boot that combines environmental data, user inputs, and biological factors to provide gentle, actionable warnings 24-48 hours before attacks.

## Team Structure and Roles

1. Backend
2. Data processing
3. Frontend
4. Data Analysis

### Project Structure
```
Personalized_Migraine_Prediction/
├── src/main/java/com/migraineguard/
│   ├── MigraineGuardApplication.java
│   ├── controller/
│   │   ├── PredictionController.java
│   │   ├── UserController.java
│   │   └── WeatherController.java
│   ├── service/
│   │   ├── PredictionService.java
│   │   ├── DataProcessingService.java
│   │   ├── WeatherService.java
│   │   └── MLService.java
│   ├── model/
│   │   ├── UserProfile.java
│   │   ├── MigrainePrediction.java
│   │   ├── DailyCheckin.java
│   │   ├── RiskAssessment.java
│   │   └── WeatherData.java
│   ├── repository/
│   └── config/
│       └── AppConfig.java
├── src/main/resources/
│   ├── application.properties
│   ├── static/
│   └── templates/
├── data/
│   ├── synthetic/
│   └── mbrain21/
├── ml/
│   ├── model_training.py (if using Python bridge)
│   └── trained_models/
└── pom.xml
```

## Data Models:

```
public enum Gender {
    MALE, FEMALE, OTHER
}

public enum RiskLevel {
    GREEN(0, 20, "Low risk - Continue as normal"),
    YELLOW(20, 50, "Moderate risk - Be mindful"), 
    ORANGE(50, 70, "High risk - Take precautions"),
    RED(70, 100, "Very high risk - Immediate action");
    
    private final int min;
    private final int max;
    private final String message;
    
    // constructor, getters
}

@Data
public class UserProfile {
    private String id;
    private Gender gender;
    private List<String> knownTriggers;
    private List<String> typicalSymptoms;
    private LocalDate birthDate;
    private Map<String, Double> triggerWeights;
}

@Data
public class MigrainePrediction {
    private RiskLevel riskLevel;
    private double probability;
    private List<String> contributingFactors;
    private String recommendation;
    private LocalDateTime predictedTime;
    private String predictedIntensity; // LIGHT, MEDIUM, STRONG
}
```

## Key REST Endpoint:

```
@RestController
@RequestMapping("/api")
public class PredictionController {
    
    @PostMapping("/predict")
    public ResponseEntity<MigrainePrediction> predictRisk(
            @RequestBody PredictionRequest request) {
        // ML prediction logic
    }
    
    @GetMapping("/user/{id}/risk")
    public ResponseEntity<RiskAssessment> getCurrentRisk(@PathVariable String id) {
        // Current risk calculation
    }
    
    @PostMapping("/checkin")
    public ResponseEntity<Void> submitDailyCheckin(@RequestBody DailyCheckin checkin) {
        // Store daily user data
    }
    
    @GetMapping("/weather-risk")
    public ResponseEntity<WeatherRisk> getWeatherRisk(
            @RequestParam double lat, 
            @RequestParam double lon) {
        // Environmental risk factors
    }
}
```

## Feature:

``` 
@Service
public class FeatureCalculator {
    
    public Map<String, Double> calculateFeatures(UserProfile user, 
                                               DailyCheckin checkin,
                                               WeatherData weather) {
        Map<String, Double> features = new HashMap<>();
        
        // Environmental features
        features.put("pressureChange", calculatePressureChange(weather));
        features.put("temperatureVariation", calculateTempVariation(weather));
        
        // Personal features
        features.put("stressScore", checkin.getStressLevel() * 0.3);
        features.put("sleepQuality", (checkin.getSleepHours() / 8.0) * 0.25);
        features.put("dietaryRisk", calculateDietaryRisk(checkin.getFoodTriggers()));
        
        // Gender-specific features
        if (user.getGender() == Gender.FEMALE) {
            features.put("hormonalRisk", calculateHormonalRisk(user));
            features.put("menstrualCycleRisk", calculateCycleRisk(user.getCycleDay()));
        }
        
        return features;
    }
}
```

# Implementation Phases
### Phase1: Project Setup and Data Analysis

Tasks:

- Initialize Spring Boot project
- Set up Maven dependencies
- Create data models
- Analyze synthetic dataset structure
- Design database schema

commands:
```
spring init --dependencies=web,data-jpa,lombok,validation Personalize_Migraine_Prediction
cd Personalize_Migraine_Predicition
```

### Phase2: Backend Development

Task:

- Implement REST controllers
- Create Service layer
- Integrate SMHI Weather API
- Build data processing pipeline
- Set up CSV data loader

### Phase3: ML Integration

Task:

 - Choose ML approach (Java libs vs Python bridge)
 - Implement feature
 - Build prediction service
 - Create risk scoring engine

### Phase4: Frontend and Demo prep

Task:

- Develop Thymeleaf templates
- Create Risk Visualization dashboard
- Implement notification system
- Preapre demo scenarios
- Test end to end flow

# ML Implementaion options

### Option A:  Pure Java ML (Tribou)

```
// Using Tribou library
DataSet dataset = DataSet.fromCSV("synthetic_data.csv");
Classifier classifier = new RandomForest();
classifier.train(dataset);
double prediction = classifier.predict(newInstance);
```

### Option B : Weka Integration
```
// Using Weka
DataSource source = new DataSource("synthetic_data.arff");
Instances data = source.getDataSet();
RandomForest forest = new RandomForest();
forest.buildClassifier(data);
```

### Option C: Python Bridge 
```
@Service
public class MLService {
    
    public MigrainePrediction predictWithPython(UserInput input) {
        ObjectMapper mapper = new ObjectMapper();
        String inputJson = mapper.writeValueAsString(input);
        
        ProcessBuilder pb = new ProcessBuilder("python3", "ml_model.py", inputJson);
        Process process = pb.start();
        
        BufferedReader reader = new BufferedReader(
            new InputStreamReader(process.getInputStream()));
        String predictionJson = reader.readLine();
        
        return mapper.readValue(predictionJson, MigrainePrediction.class);
    }
}
```

# Demo Scenarios

### Scenario 1: Female User (Hormonal Factors)

```
INPUT: 
  - Gender: FEMALE
  - Cycle Day: 12 (ovulation)
  - Stress: High (8/10)
  - Sleep: 5 hours
  - Weather: Pressure dropping

PREDICTION:
  🟠 Risk Level: ORANGE (65%)
  ⚡ Intensity: MEDIUM-STRONG
  💡 Factors: Hormonal (40%), Stress (35%), Weather (25%)

RECOMMENDATION:
  "High risk detected. Consider taking preventive medication 
   and rescheduling demanding activities tomorrow."

```

### Scenario 2: Male User (Stress Triggers)

```
INPUT:
  - Gender: MALE  
  - Stress: Extreme (9/10)
  - Sleep: 4 hours
  - Work Deadline: Yes

PREDICTION:
  🟡 Risk Level: YELLOW (45%)
  ⚡ Intensity: LIGHT-MEDIUM
  💡 Factors: Stress (70%), Sleep (30%)

RECOMMENDATION:
  "Moderate risk detected. Try to take breaks and aim for 
   better sleep tonight to reduce migraine probability."

```

# Work flow Diagram

```
User Input → Spring Controller → Feature Service → ML Service → Risk Engine → Notification Service
    ↓              ↓               ↓              ↓             ↓             ↓
Daily Checkin   Validation     Calculation   Prediction    Color Coding   Gentle Alert

```