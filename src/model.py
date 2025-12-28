import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, MaxPooling1D, 
    Flatten, Concatenate, Dropout, BatchNormalization, 
    GlobalAveragePooling2D, GlobalAveragePooling1D, Reshape, Multiply, Conv2D,
    Activation, RandomFlip, RandomRotation
)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model

def attention_block(inputs, name="attention"):
    # Attention to filter noise
    a = Dense(1, activation='tanh', use_bias=True)(inputs)
    a = Flatten()(a)
    a = Activation('softmax')(a)
    a = Reshape((inputs.shape[1], 1))(a)
    return Multiply()([inputs, a])

def build_multimodal_model(shape_clin, shape_ctg, shape_act, shape_img=(128, 128, 1)):
    
    # --- ENCODERS (Feature Extractors) ---
    
    # 1. Clinical Encoder
    input_clin = Input(shape=shape_clin, name='input_clinical')
    x_clin = Dense(128, activation='relu')(input_clin)
    x_clin = BatchNormalization()(x_clin)
    feat_clin = Dense(64, activation='relu')(x_clin) # Shared clinical features
    
    # 2. CTG Encoder
    input_ctg = Input(shape=shape_ctg, name='input_ctg')
    x_ctg = Conv1D(64, 3, padding='same', activation='relu')(input_ctg)
    x_ctg = MaxPooling1D(2)(x_ctg)
    x_ctg = LSTM(64, return_sequences=True)(x_ctg)
    x_ctg = attention_block(x_ctg)
    feat_ctg = GlobalAveragePooling1D()(x_ctg)
    
    # 3. Activity Encoder
    input_act = Input(shape=shape_act, name='input_activity')
    x_act = LSTM(64, return_sequences=True)(input_act)
    x_act = attention_block(x_act)
    feat_act = GlobalAveragePooling1D()(x_act)

    # 4. Image Encoder (MobileNetV2)
    input_img = Input(shape=shape_img, name='input_image')
    x_img_aug = RandomFlip("horizontal")(input_img)
    x_img_rgb = Conv2D(3, (3, 3), padding='same')(x_img_aug) # 1ch -> 3ch
    
    base_cnn = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_cnn.trainable = False # Freeze again to stabilize weight prediction
    
    x_img_feat = base_cnn(x_img_rgb)
    feat_img = GlobalAveragePooling2D()(x_img_feat)
    
    # --- TOWER 1: RISK SPECIALIST ---
    # Fuses Clinical + CTG + Activity (Physiological Data)
    # We purposefully EXCLUDE images here if they were adding noise to risk
    risk_fusion = Concatenate()([feat_clin, feat_ctg, feat_act])
    
    r = Dense(64, activation='relu')(risk_fusion)
    r = BatchNormalization()(r)
    r = Dropout(0.3)(r)
    # Clinical Highway for Risk (Direct connection)
    r_highway = Concatenate()([r, input_clin]) 
    
    output_risk = Dense(3, activation='softmax', name='output_risk')(r_highway)
    
    # --- TOWER 2: WEIGHT SPECIALIST ---
    # Fuses Clinical + Images (Physical Growth Data)
    # Weight is mostly about "Mother's Health + Baby's Size"
    weight_fusion = Concatenate()([feat_clin, feat_img])
    
    w = Dense(64, activation='relu')(weight_fusion)
    w = Dense(32, activation='relu')(w)
    
    output_weight = Dense(1, activation='linear', name='output_weight')(w)
    
    # --- MODEL ---
    model = Model(
        inputs=[input_clin, input_ctg, input_act, input_img], 
        outputs=[output_risk, output_weight]
    )
    
    # Reset Learning Rate to standard 0.001 since we froze CNN again
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=opt, 
        loss={'output_risk': 'sparse_categorical_crossentropy', 'output_weight': 'mae'},
        # BALANCED WEIGHTS: Don't let Risk bully Weight
        loss_weights={'output_risk': 1.0, 'output_weight': 1.0},
        metrics={'output_risk': 'accuracy', 'output_weight': 'mae'}
    )
    
    return model

if __name__ == "__main__":
    model = build_multimodal_model((18,), (11, 1), (50, 3), (128, 128, 1))
    model.summary()
    print("✅ Decoupled Twin-Tower Model Built")