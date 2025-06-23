import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
import pickle

# Configuration
image_size = (150, 150)
batch_size = 64
epochs = 500
initial_lr = 1e-5

def load_and_verify_data():
    """Load dataset with visual verification"""
    train_ds = keras.utils.image_dataset_from_directory(
        "data",
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        "data",
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )
    
    # Visual verification
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(train_ds.class_names[np.argmax(labels[i])])
            plt.axis("off")
        plt.savefig('data_samples.png')
        plt.close()
    
    return train_ds, val_ds, train_ds.class_names

def compute_class_weights_tf(dataset, num_classes):
    """Pure TF implementation of class weights"""
    class_counts = tf.zeros(num_classes, dtype=tf.int32)
    
    for _, labels in dataset:
        class_counts += tf.reduce_sum(
            tf.cast(labels, tf.int32),
            axis=0
        )
    
    total_samples = tf.reduce_sum(class_counts)
    class_weights = total_samples / (num_classes * (class_counts + 1))
    return tf.linalg.normalize(class_weights, ord=1)[0].numpy()

def build_model(input_shape, num_classes):
    """Optimized model architecture"""
    inputs = keras.Input(shape=input_shape)
    
    # Moderate augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomBrightness(0.1)(x)
    x = layers.Rescaling(1./255)(x)
    
    # Conv blocks
    x = layers.Conv2D(32, 3, padding='same', activation='relu', 
                     kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(64, 3, padding='same', activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(128, 3, padding='same', activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    
    x = layers.Conv2D(256, 3, padding='same', activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classifier head
    x = layers.Dense(512, activation='relu',
                    kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    optimizer = keras.optimizers.Adam(
        learning_rate=initial_lr,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')]
    )
    
    return model

def train_model(model, train_ds, val_ds, class_weights):
    """Training with enhanced callbacks"""
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.keras",
            monitor='val_auc',
            save_best_only=True,
            mode='max'
        ),
        #keras.callbacks.EarlyStopping(
        #    monitor='val_auc',
        #    patience=15,
        #    restore_best_weights=True
        #),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        ),
        keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        class_weight=dict(enumerate(class_weights))
    )
    
    return history

def analyze_results(history):
    """Training performance analysis"""
    plt.figure(figsize=(18, 12))
    
    metrics = ['accuracy', 'auc', 'precision', 'recall']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        plt.title(metric.upper())
        plt.legend()
    
    plt.savefig('training_metrics.png')
    plt.close()
    
    print("\nTraining Report:")
    print(f"Best Val Accuracy: {max(history.history['val_accuracy']):.2%}")
    print(f"Best Val AUC: {max(history.history['val_auc']):.3f}")

def main():
    # Load data
    train_ds, val_ds, class_names = load_and_verify_data()
    print(f"Classes: {class_names}")
    
    # Save class names
    with open('class_names.pkl', 'wb') as f:
        pickle.dump(class_names, f)
    
    # Compute class weights
    class_weights = compute_class_weights_tf(train_ds, len(class_names))
    print("Class weights:", class_weights)
    
    # Build model
    model = build_model(
        input_shape=image_size + (3,),
        num_classes=len(class_names)
    )
    model.summary()
    
    # Train
    history = train_model(model, train_ds, val_ds, class_weights)
    
    # Save final model
    model.save("fruit_classifier.keras")
    
    # Analyze
    analyze_results(history)

    # ====== TESTING SECTION ======
    # Load and preprocess test image
    test_image_path = "data/banana/Banana096.png"  # <-- REPLACE THIS with your image path
    try:
        img = keras.utils.load_img(test_image_path, target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = keras.ops.expand_dims(img_array, 0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize like in training

        # Predict
        predictions = model.predict(img_array)
        probabilities = keras.ops.softmax(predictions[0])  # Convert logits to probabilities

        # Get results
        predicted_index = np.argmax(probabilities)
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100

        # Display results
        print("\n=== TEST RESULTS ===")
        print(f"Test Image: {test_image_path}")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        
        # Print all class probabilities
        print("\nClass Probabilities:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}: {probabilities[i]*100:.2f}%")
            
        # Display the test image
        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"\nError testing image: {str(e)}")
    

if __name__ == "__main__":
    main()
