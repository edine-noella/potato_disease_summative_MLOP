"""
Locust Load Testing for Potato Disease Classification API
Simulates flood of requests to test model performance and latency
"""

from locust import HttpUser, task, between
import random
import os
from io import BytesIO

class DiseaseClassifierUser(HttpUser):
    """Simulates a user making prediction requests"""
    
    # Wait time between requests (1-3 seconds)
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when a user starts"""
        print("User started - preparing test image")
        # In real scenario, you would load actual test images
        # For now, we'll test other endpoints
    
    @task(3)
    def health_check(self):
        """Test health check endpoint - most frequent"""
        with self.client.get("/api/monitor/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(2)
    def get_uptime(self):
        """Test uptime endpoint"""
        with self.client.get("/api/monitor/uptime", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Uptime check failed with status {response.status_code}")
    
    @task(2)
    def get_stats(self):
        """Test stats endpoint"""
        with self.client.get("/api/monitor/stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Stats check failed with status {response.status_code}")
    
    @task(1)
    def get_data_stats(self):
        """Test data statistics endpoint"""
        with self.client.get("/api/data/stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Data stats failed with status {response.status_code}")
    
    @task(1)
    def get_training_status(self):
        """Test training status endpoint"""
        with self.client.get("/api/train/status", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Training status failed with status {response.status_code}")
    
    @task(1)
    def predict_with_dummy_image(self):
        """
        Test prediction endpoint with a dummy image
        Note: This will likely fail without a real image, but tests the endpoint
        """
        # Create a dummy file
        dummy_image = BytesIO(b"fake image data")
        
        files = {'file': ('test.jpg', dummy_image, 'image/jpeg')}
        
        with self.client.post("/api/predict", files=files, catch_response=True) as response:
            if response.status_code in [200, 400]:  # 400 is expected for invalid image
                response.success()
            else:
                response.failure(f"Prediction failed with status {response.status_code}")


class HeavyLoadUser(HttpUser):
    """Simulates heavy load - rapid fire requests"""
    
    wait_time = between(0.1, 0.5)  # Much faster requests
    
    @task
    def rapid_health_checks(self):
        """Rapid health checks to test system under load"""
        self.client.get("/api/monitor/health")


# Advanced load test with different user behaviors
class DashboardUser(HttpUser):
    """Simulates a user browsing the dashboard"""
    
    wait_time = between(2, 5)
    
    @task(3)
    def view_dashboard(self):
        """Load dashboard page"""
        self.client.get("/dashboard")
    
    @task(2)
    def view_visualizations(self):
        """Load visualizations page"""
        self.client.get("/visualizations")
    
    @task(1)
    def view_home(self):
        """Load home page"""
        self.client.get("/")
    
    @task(2)
    def check_monitoring(self):
        """Check monitoring endpoints"""
        self.client.get("/api/monitor/uptime")
        self.client.get("/api/monitor/stats")


# For testing with real images
class RealImagePredictionUser(HttpUser):
    """
    User that makes predictions with real images
    Place test images in tests/sample_images/ directory
    """
    
    wait_time = between(2, 5)
    
    def on_start(self):
        """Load sample images"""
        self.sample_images_dir = "tests/sample_images"
        if os.path.exists(self.sample_images_dir):
            self.images = [f for f in os.listdir(self.sample_images_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
        else:
            self.images = []
            print(f"Warning: {self.sample_images_dir} not found")
    
    @task
    def predict_real_image(self):
        """Make prediction with a real image"""
        if not self.images:
            return
        
        # Select random image
        image_name = random.choice(self.images)
        image_path = os.path.join(self.sample_images_dir, image_name)
        
        with open(image_path, 'rb') as img:
            files = {'file': (image_name, img, 'image/jpeg')}
            
            with self.client.post("/api/predict", files=files, catch_response=True) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data.get('success'):
                            response.success()
                        else:
                            response.failure("Prediction unsuccessful")
                    except:
                        response.failure("Invalid JSON response")
                else:
                    response.failure(f"Prediction failed with status {response.status_code}")


if __name__ == "__main__":
    print("""
    ================================================================================
    LOCUST LOAD TESTING FOR POTATO DISEASE CLASSIFIER
    ================================================================================
    
    To run load tests:
    
    1. Start the Flask API:
       python src/api.py
    
    2. Run Locust:
       locust -f tests/locustfile.py --host=http://localhost:5000
    
    3. Open browser to http://localhost:8089
    
    4. Configure test:
       - Number of users (e.g., 100)
       - Spawn rate (e.g., 10 users/second)
       - Host: http://localhost:5000
    
    5. Test scenarios:
       - Light load: 10-50 users
       - Medium load: 50-200 users
       - Heavy load: 200-1000 users
    
    6. Monitor:
       - Response times
       - Requests per second
       - Failure rates
    
    User Types:
    - DiseaseClassifierUser: Normal API usage
    - HeavyLoadUser: Stress testing
    - DashboardUser: UI browsing
    - RealImagePredictionUser: Prediction with real images (requires sample images)
    
    ================================================================================
    """)