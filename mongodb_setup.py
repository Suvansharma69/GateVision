from pymongo import MongoClient
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB_NAME = "license_plate_db"
MONGO_COLLECTION_NAME = "plates"

# Enhanced sample dataset with more realistic data
sample_plates = [
    {
        "plate_number": "ABC1234",
        "timestamp": datetime(2025, 7, 31, 12, 0, 0),
        "confidence": 95.5,
        "screenshot_path": "D:\\vehicle_speed\\no_plate\\screenshots\\HD_plate_JKL7890_20250731_113015_654.jpg",
        "created_at": datetime(2025, 7, 31, 11, 30, 15),
        "detection_id": "JKL7890_20250731_113015_654",
        "file_size_kb": 189.3,
        "enhancement_applied": True,
        "detection_method": "YOLO + OCR",
        "image_quality": "Ultra HD Enhanced"
    },
    {
        "plate_number": "MNO2468",
        "timestamp": datetime(2025, 7, 30, 16, 22, 45),
        "confidence": 94.1,
        "screenshot_path": "D:\\vehicle_speed\\no_plate\\screenshots\\HD_plate_MNO2468_20250730_162245_987.jpg",
        "created_at": datetime(2025, 7, 30, 16, 22, 45),
        "detection_id": "MNO2468_20250730_162245_987",
        "file_size_kb": 267.9,
        "enhancement_applied": True,
        "detection_method": "YOLO + OCR",
        "image_quality": "Ultra HD Enhanced"
    }
]

def populate_mongodb():
    try:
        # Connect to MongoDB
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        logger.info("✅ Successfully connected to MongoDB")

        # Access database and collection
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]

        # Check if collection already has data
        existing_count = collection.count_documents({})
        logger.info(f"📊 Found {existing_count} existing records in {MONGO_COLLECTION_NAME} collection")

        # Clear existing data (optional - remove this if you want to keep existing records)
        if existing_count > 0:
            response = input("Do you want to clear existing data? (y/N): ").lower().strip()
            if response == 'y' or response == 'yes':
                collection.delete_many({})
                logger.info("🔄 Cleared existing data in plates collection")
            else:
                logger.info("📝 Keeping existing data, adding sample records")

        # Insert sample data
        result = collection.insert_many(sample_plates)
        logger.info(f"📊 Inserted {len(result.inserted_ids)} sample records into {MONGO_COLLECTION_NAME} collection")

        # Create indexes for better performance
        logger.info("🔧 Creating database indexes for optimal performance...")
        
        # Create indexes
        collection.create_index("plate_number")
        collection.create_index("timestamp")
        collection.create_index("created_at")
        collection.create_index("confidence")
        collection.create_index("detection_id", unique=True)
        
        logger.info("✅ Database indexes created successfully")

        # Display collection statistics
        total_records = collection.count_documents({})
        unique_plates = len(collection.distinct("plate_number"))
        
        # Recent records (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        recent_records = collection.count_documents({
            "timestamp": {"$gte": yesterday}
        })

        logger.info("\n" + "="*50)
        logger.info("📊 MONGODB COLLECTION STATISTICS")
        logger.info("="*50)
        logger.info(f"Database: {MONGO_DB_NAME}")
        logger.info(f"Collection: {MONGO_COLLECTION_NAME}")
        logger.info(f"Total Records: {total_records}")
        logger.info(f"Unique Plates: {unique_plates}")
        logger.info(f"Recent Records (24h): {recent_records}")
        logger.info("="*50)

        # Show sample queries
        logger.info("\n🔍 SAMPLE SEARCH QUERIES:")
        logger.info("-" * 30)
        
        # Search for a specific plate
        sample_plate = collection.find_one({"plate_number": "ABC1234"})
        if sample_plate:
            logger.info(f"Found plate ABC1234: Confidence {sample_plate['confidence']}%, Captured {sample_plate['timestamp']}")
        
        # Show recent plates
        recent_plates = list(collection.find().sort("timestamp", -1).limit(3))
        logger.info(f"Most recent {len(recent_plates)} plates:")
        for plate in recent_plates:
            logger.info(f"  - {plate['plate_number']} ({plate['confidence']}%) at {plate['timestamp']}")

        logger.info("\n✅ MongoDB setup completed successfully!")
        logger.info("💡 You can now:")
        logger.info("   1. Start the license plate detection system")
        logger.info("   2. Search plates using the web dashboard")
        logger.info("   3. Use MongoDB Compass to view/manage data")
        logger.info(f"   4. Connect to: {MONGO_URI}")

    except Exception as e:
        logger.error(f"❌ Error populating MongoDB: {e}")
        logger.error("💡 Troubleshooting tips:")
        logger.error("   1. Make sure MongoDB is running: mongod")
        logger.error("   2. Check if MongoDB service is started")
        logger.error("   3. Verify connection string: " + MONGO_URI)
        logger.error("   4. Install pymongo: pip install pymongo")
    finally:
        if 'client' in locals():
            client.close()
            logger.info("🔌 MongoDB connection closed")

def test_mongodb_connection():
    """Test MongoDB connection and basic operations"""
    try:
        logger.info("🔍 Testing MongoDB connection...")
        
        # Connect to MongoDB
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        logger.info("✅ MongoDB connection test successful")
        
        # Test database access
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        
        # Test write operation
        test_doc = {
            "test": True,
            "timestamp": datetime.now(),
            "message": "Connection test successful"
        }
        
        result = collection.insert_one(test_doc)
        logger.info(f"✅ Write test successful, inserted document with ID: {result.inserted_id}")
        
        # Test read operation
        found_doc = collection.find_one({"_id": result.inserted_id})
        if found_doc:
            logger.info("✅ Read test successful")
        
        # Clean up test document
        collection.delete_one({"_id": result.inserted_id})
        logger.info("✅ Cleanup successful")
        
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ MongoDB connection test failed: {e}")
        return False

def show_collection_info():
    """Display information about the existing collection"""
    try:
        client = MongoClient(MONGO_URI)
        client.admin.command('ping')
        
        db = client[MONGO_DB_NAME]
        collection = db[MONGO_COLLECTION_NAME]
        
        # Get collection stats
        total_docs = collection.count_documents({})
        
        if total_docs == 0:
            logger.info("📊 Collection is empty")
            return
        
        logger.info(f"📊 Collection '{MONGO_COLLECTION_NAME}' contains {total_docs} documents")
        
        # Show sample documents
        sample_docs = list(collection.find().limit(5))
        logger.info("\n📋 Sample documents:")
        for i, doc in enumerate(sample_docs, 1):
            logger.info(f"  {i}. Plate: {doc.get('plate_number', 'N/A')}, "
                       f"Confidence: {doc.get('confidence', 'N/A')}%, "
                       f"Time: {doc.get('timestamp', 'N/A')}")
        
        # Show indexes
        indexes = collection.list_indexes()
        logger.info("\n🔧 Existing indexes:")
        for index in indexes:
            logger.info(f"  - {index['name']}: {list(index['key'].keys())}")
        
        client.close()
        
    except Exception as e:
        logger.error(f"❌ Error getting collection info: {e}")

if __name__ == "__main__":
    print("🚀 MongoDB License Plate Database Setup")
    print("=" * 40)
    
    # Test connection first
    if not test_mongodb_connection():
        print("\n❌ MongoDB connection failed. Please ensure MongoDB is running.")
        print("💡 Start MongoDB with: mongod --dbpath /path/to/your/db")
        exit(1)
    
    print("\n" + "="*40)
    print("Choose an option:")
    print("1. Populate database with sample data")
    print("2. Show existing collection info")
    print("3. Test connection only")
    print("="*40)
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == '1':
        populate_mongodb()
    elif choice == '2':
        show_collection_info()
    elif choice == '3':
        print("✅ Connection test completed successfully!")
else:
    print("❌ Invalid choice. Please run the script again.")