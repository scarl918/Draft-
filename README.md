/* Enhanced model loading with type-specific configuration */
ml_model_t* ml_load_model_enhanced(ml_framework_t* fw, const ml_model_config_enhanced_t* config) {
    if (!fw || !config) return NULL;
    
    /* Create model with specific type */
    ml_model_t* model = ml_create_model(config->model_type);
    if (!model) return NULL;
    
    /* Set basic info */
    model->name = strdup(config->model_name);
    model->path = strdup(config->model_path);
    model->description = config->description ? strdup(config->description) : NULL;
    model->version = strdup("1.0.0");  /* Default version */
    
    /* Create session options with model-specific optimizations */
    OrtSessionOptions* session_options;
    fw->ort_api->CreateSessionOptions(&session_options);
    
    /* Apply performance hints */
    if (config->perf_hints) {
        if (config->perf_hints->prefer_gpu) {
            /* TODO: Add GPU provider */
        }
        
        if (config->perf_hints->num_threads > 0) {
            fw->ort_api->SetIntraOpNumThreads(session_options, config->perf_hints->num_threads);
        }
        
        model->perf_hints = *config->perf_hints;
    }
    
    /* Model type specific optimizations */
    switch (config->model_type) {
        case MODEL_TYPE_RANDOM_FOREST:
        case MODEL_TYPE_XGBOOST:
            /* Tree-based models work better with sequential execution */
            fw->ort_api->SetSessionExecutionMode(session_options, ORT_SEQUENTIAL);
            break;
            
        case MODEL_TYPE_NEURAL_NETWORK:
            /* Neural networks benefit from parallel execution */
            fw->ort_api->SetSessionExecutionMode(session_options, ORT_PARALLEL);
            fw->ort_api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);
            break;
            
        default:
            fw->ort_api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED);
            break;
    }
    
    /* Create session */
    OrtStatus* status = fw->ort_api->CreateSession(fw->ort_env, config->model_path, 
                                                  session_options, &model->session);
    
    if (status != NULL) {
        const char* error_msg = fw->ort_api->GetErrorMessage(status);
        fprintf(stderr, "Failed to load %s model '%s': %s\n", 
                model_type_to_string(config->model_type), config->model_name, error_msg);
        fw->ort_api->ReleaseStatus(status);
        fw->ort_api->ReleaseSessionOptions(session_options);
        ml_destroy_model(model);
        return NULL;
    }
    
    model->session_options = session_options;
    
    /* Get input/output info from session */
    size_t num_inputs;
    fw->ort_api->SessionGetInputCount(model->session, &num_inputs);
    model->num_inputs = num_inputs;
    model->inputs = calloc(num_inputs, sizeof(tensor_info_t));
    
    /* Populate input tensor info */
    for (size_t i = 0; i < num_inputs; i++) {
        char* input_name;
        OrtAllocator* allocator;
        fw->ort_api->GetAllocatorWithDefaultOptions(&allocator);
        fw->ort_api->SessionGetInputName(model->session, i, allocator, &input_name);
        
        OrtTypeInfo* type_info;
        fw->ort_api->SessionGetInputTypeInfo(model->session, i, &type_info);
        
        const OrtTensorTypeAndShapeInfo* tensor_info;
        fw->ort_api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
        
        size_t num_dims;
        fw->ort_api->GetDimensionsCount(tensor_info, &num_dims);
        
        int64_t* dims = malloc(num_dims * sizeof(int64_t));
        fw->ort_api->GetDimensions(tensor_info, dims, num_dims);
        
        ONNXTensorElementDataType dtype;
        fw->ort_api->GetTensorElementType(tensor_info, &dtype);
        
        model->inputs[i] = *create_tensor_info(input_name, dtype, dims, num_dims);
        
        free(dims);
        fw->ort_api->ReleaseTypeInfo(type_info);
        allocator->Free(allocator, input_name);
    }
    
    /* Similar for outputs */
    size_t num_outputs;
    fw->ort_api->SessionGetOutputCount(model->session, &num_outputs);
    model->num_outputs = num_outputs;
    model->outputs = calloc(num_outputs, sizeof(tensor_info_t));
    
    /* Copy preprocessing config if provided */
    if (config->preprocess_config) {
        model->preprocess = *config->preprocess_config;
        /* Deep copy arrays if needed */
        if (config->preprocess_config->input_mean) {
            size_t feature_count = model->inputs[0].total_elements;
            model->preprocess.input_mean = malloc(feature_count * sizeof(float));
            memcpy(model->preprocess.input_mean, config->preprocess_config->input_mean, 
                   feature_count * sizeof(float));
        }
        if (config->preprocess_config->input_std) {
            size_t feature_count = model->inputs[0].total_elements;
            model->preprocess.input_std = malloc(feature_count * sizeof(float));
            memcpy(model->preprocess.input_std, config->preprocess_config->input_std, 
                   feature_count * sizeof(float));
        }
        if (config->preprocess_config->input_min) {
            size_t feature_count = model->inputs[0].total_elements;
            model->preprocess.input_min = malloc(feature_count * sizeof(float));
            memcpy(model->preprocess.input_min, config->preprocess_config->input_min, 
                   feature_count * sizeof(float));
        }
        if (config->preprocess_config->input_max) {
            size_t feature_count = model->inputs[0].total_elements;
            model->preprocess.input_max = malloc(feature_count * sizeof(float));
            memcpy(model->preprocess.input_max, config->preprocess_config->input_max, 
                   feature_count * sizeof(float));
        }
    }
    
    /* Set model-specific metadata based on type */
    switch (config->model_type) {
        case MODEL_TYPE_RANDOM_FOREST:
            model->metadata.rf_metadata.num_trees = 100;  /* Default, should come from config */
            model->metadata.rf_metadata.max_depth = 10;
            model->metadata.rf_metadata.num_features = model->inputs[0].total_elements;
            break;
            
        case MODEL_TYPE_KNN:
            model->metadata.knn_metadata.k_value = 5;  /* Default */
            model->metadata.knn_metadata.distance_metric = strdup("euclidean");
            break;
            
        case MODEL_TYPE_NEURAL_NETWORK:
            model->metadata.nn_metadata.supports_dynamic_batch = true;
            model->metadata.nn_metadata.num_layers = 5;  /* Should be detected from model */
            model->metadata.nn_metadata.activation_function = strdup("relu");
            break;
    }
    
    /* Set default batch size if not specified */
    model->max_batch_size = 1;
    if (config->model_type == MODEL_TYPE_NEURAL_NETWORK) {
        model->max_batch_size = 32;  /* NNs typically support batching */
        model->supports_batching = true;
    }
    
    /* Add to registry */
    ml_registry_add_model(fw->registry, model);
    
    model->is_loaded = true;
    
    printf("Loaded %s model '%s' with %zu inputs and %zu outputs\n",
           model_type_to_string(config->model_type), model->name, 
           model->num_inputs, model->num_outputs);
    
    return model;
}

/* Original ml_load_model for backward compatibility */
ml_model_t* ml_load_model(ml_framework_t* fw, const ml_model_config_t* config) {
    ml_model_config_enhanced_t enhanced_config = {
        .model_path = config->model_path,
        .model_name = config->model_name,
        .model_type = MODEL_TYPE_CUSTOM,  /* Default type */
        .description = "Legacy model",
        .num_inputs = 1,
        .num_outputs = 1,
        .preprocess_config = NULL,
        .perf_hints = NULL
    };
    
    return ml_load_model_enhanced(fw, &enhanced_config);
}

void ml_unload_model(ml_framework_t* fw, ml_model_t* model) {
    if (!fw || !model) return;
    
    /* Remove from registry */
    pthread_rwlock_wrlock(&fw->registry->lock);
    for (size_t i = 0; i < fw->registry->count; i++) {
        if (fw->registry->models[i] == model) {
            /* Shift remaining models */
            for (size_t j = i; j < fw->registry->count - 1; j++) {
                fw->registry->models[j] = fw->registry->models[j + 1];
            }
            fw->registry->count--;
            break;
        }
    }
    pthread_rwlock_unlock(&fw->registry->lock);
    
    /* Release ONNX session */
    if (model->session) fw->ort_api->ReleaseSession(model->session);
    if (model->session_options) fw->ort_api->ReleaseSessionOptions(model->session_options);
    
    ml_destroy_model(model);
}

ml_model_t* ml_get_model(ml_framework_t* fw, const char* model_name) {
    if (!fw || !model_name) return NULL;
    return ml_registry_get_model(fw->registry, model_name);
}

/* Enhanced async inference with preprocessing */
uint64_t ml_inference_async(
    ml_framework_t* fw,
    ml_model_t* model,
    const void* input,
    size_t input_size,
    inference_callback_fn callback,
    void* user_data,
    ml_priority_t priority,
    uint32_t timeout_ms
) {
    if (!fw || !model || !input) return 0;
    
    /* Allocate request */
    ml_async_request_t* req = calloc(1, sizeof(ml_async_request_t));
    if (!req) return 0;
    
    /* Generate unique request ID */
    pthread_mutex_lock(&fw->id_lock);
    req->request_id = ++fw->next_request_id;
    pthread_mutex_unlock(&fw->id_lock);
    
    /* Get PP core ID */
    req->pp_core_id = sched_getcpu() % fw->num_channels;
    
    /* Allocate buffers */
    req->input_buffer = ml_alloc_input_buffer(fw, input_size);
    req->output_buffer = ml_alloc_output_buffer(fw, model->outputs[0].total_elements * sizeof(float));
    
    if (!req->input_buffer || !req->output_buffer) {
        if (req->input_buffer) ml_free_input_buffer(fw, req->input_buffer);
        if (req->output_buffer) ml_free_output_buffer(fw, req->output_buffer);
        free(req);
        return 0;
    }
    
    /* Apply preprocessing if needed */
    if (model->preprocess.type != PREPROCESS_NONE) {
        ml_preprocess_input(model, input, req->input_buffer, input_size);
    } else {
        memcpy(req->input_buffer, input, input_size);
    }
    
    /* Fill request */
    req->model = model;
    req->input_size = input_size;
    req->output_size = model->outputs[0].total_elements * sizeof(float);
    req->callback = callback;
    req->user_data = user_data;
    req->status = ML_STATUS_PENDING;
    
    /* Create run options with timeout and priority */
    fw->ort_api->CreateRunOptions(&req->run_options);
    
    if (timeout_ms > 0) {
        char timeout_str[32];
        snprintf(timeout_str, sizeof(timeout_str), "%u", timeout_ms);
        fw->ort_api->AddRunConfigEntry(req->run_options, "run_timeout", timeout_str);
    }
    
    /* Set priority based on model type and request priority */
    if (model->type == MODEL_TYPE_NEURAL_NETWORK && priority == ML_PRIORITY_HIGH) {
        fw->ort_api->AddRunConfigEntry(req->run_options, "priority", "high");
        fw->ort_api->AddRunConfigEntry(req->run_options, "use_gpu", "1");
    } else if (priority == ML_PRIORITY_LOW) {
        fw->ort_api->AddRunConfigEntry(req->run_options, "priority", "low");
    }
    
    /* Create input tensor with proper shape */
    int64_t* input_shape = model->inputs[0].shape;
    size_t num_dims = model->inputs[0].num_dimensions;
    
    /* Handle dynamic batch dimension */
    int64_t* actual_shape = malloc(num_dims * sizeof(int64_t));
    memcpy(actual_shape, input_shape, num_dims * sizeof(int64_t));
    if (input_shape[0] == -1) {
        actual_shape[0] = 1;  /* Single sample */
    }
    
    OrtStatus* status = fw->ort_api->CreateTensorWithDataAsOrtValue(
        fw->memory_info, req->input_buffer, input_size,
        actual_shape, num_dims, model->inputs[0].dtype,
        &req->input_tensor
    );
    
    free(actual_shape);
    
    if (status != NULL) {
        fw->ort_api->ReleaseStatus(status);
        fw->ort_api->ReleaseRunOptions(req->run_options);
        ml_free_input_buffer(fw, req->input_buffer);
        ml_free_output_buffer(fw, req->output_buffer);
        free(req);
        return 0;
    }
    
    /* Track request */
    pthread_rwlock_wrlock(&fw->requests_lock);
    fw->active_requests[req->request_id % fw->max_requests] = req;
    pthread_rwlock_unlock(&fw->requests_lock);
    
    /* Create async context */
    async_context_t* ctx = malloc(sizeof(async_context_t));
    ctx->request = req;
    ctx->framework = fw;
    ctx->start_time = get_timestamp_us();
    
    /* Get input/output names from model */
    const char* input_names[1] = {model->inputs[0].name};
    const char* output_names[1] = {model->outputs[0].name};
    
    /* Submit async inference using native ONNX RunAsync */
    status = fw->ort_api->RunAsync(
        model->session,
        req->run_options,
        input_names,
        (const OrtValue* const*)&req->input_tensor,
        1,
        output_names,
        1,
        &req->output_tensor,
        onnx_async_callback_enhanced,
        ctx
    );
    
    if (status != NULL) {
        const char* error = fw->ort_api->GetErrorMessage(status);
        fprintf(stderr, "[%s] RunAsync failed: %s\n", model_type_to_string(model->type), error);
        fw->ort_api->ReleaseStatus(status);
        
        /* Cleanup on failure */
        pthread_rwlock_wrlock(&fw->requests_lock);
        fw->active_requests[req->request_id % fw->max_requests] = NULL;
        pthread_rwlock_unlock(&fw->requests_lock);
        
        fw->ort_api->ReleaseValue(req->input_tensor);
        fw->ort_api->ReleaseRunOptions(req->run_options);
        ml_free_input_buffer(fw, req->input_buffer);
        ml_free_output_buffer(fw, req->output_buffer);
        free(ctx);
        free(req);
        return 0;
    }
    
    /* Update statistics */
    __atomic_fetch_add(&fw->stats.total_requests, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&fw->stats.requests_per_type[model->type], 1, __ATOMIC_RELAXED);
    
    return req->request_id;
}

/* Batch inference with model-specific handling */
uint64_t ml_inference_batch_async(
    ml_framework_t* fw,
    ml_model_t* model,
    const void** inputs,
    size_t* input_sizes,
    size_t batch_size,
    batch_callback_fn callback,
    void* user_data
) {
    if (!fw || !model || !inputs || batch_size == 0) return 0;
    
    /* Check if model supports batching */
    if (!model->supports_batching) {
        fprintf(stderr, "Model '%s' does not support batching\n", model->name);
        return 0;
    }
    
    /* Limit batch size to model's maximum */
    if (batch_size > model->max_batch_size) {
        batch_size = model->max_batch_size;
    }
    
    /* Calculate total size for batch buffer */
    size_t single_input_elements = model->inputs[0].total_elements;
    size_t element_size = sizeof(double);  /* Assuming double, should check dtype */
    size_t total_size = batch_size * single_input_elements * element_size;
    
    /* Allocate batch buffer */
    void* batch_buffer = ml_alloc_input_buffer(fw, total_size);
    if (!batch_buffer) return 0;
    
    /* Apply preprocessing to each sample and copy to batch buffer */
    for (size_t i = 0; i < batch_size; i++) {
        void* sample_dest = (char*)batch_buffer + i * single_input_elements * element_size;
        
        if (model->preprocess.type != PREPROCESS_NONE) {
            ml_preprocess_input(model, inputs[i], sample_dest, 
                              single_input_elements * element_size);
        } else {
            memcpy(sample_dest, inputs[i], single_input_elements * element_size);
        }
    }
    
    /* Create batch tensor with shape [batch_size, ...] */
    int64_t* batch_shape = malloc(model->inputs[0].num_dimensions * sizeof(int64_t));
    batch_shape[0] = batch_size;
    for (size_t i = 1; i < model->inputs[0].num_dimensions; i++) {
        batch_shape[i] = model->inputs[0].shape[i];
    }
    
    OrtValue* batch_tensor;
    OrtStatus* status = fw->ort_api->CreateTensorWithDataAsOrtValue(
        fw->memory_info, batch_buffer, total_size,
        batch_shape, model->inputs[0].num_dimensions, model->inputs[0].dtype,
        &batch_tensor
    );
    
    free(batch_shape);
    
    if (status != NULL) {
        fw->ort_api->ReleaseStatus(status);
        ml_free_input_buffer(fw, batch_buffer);
        return 0;
    }
    
    /* Create batch request similar to single request */
    ml_async_request_t* req = calloc(1, sizeof(ml_async_request_t));
    req->request_id = ++fw->next_request_id;
    req->model = model;
    req->input_buffer = batch_buffer;
    req->input_tensor = batch_tensor;
    /* Note: For batch, we'd need different callback handling */
    
    /* Update model stats */
    pthread_mutex_lock(&model->lock);
    model->stats.total_batches++;
    pthread_mutex_unlock(&model->lock);
    
    /* TODO: Complete batch implementation with proper callback */
    
    return req->request_id;
}

/* Request management functions */
int ml_cancel_request(ml_framework_t* fw, uint64_t request_id) {
    if (!fw) return -1;
    
    pthread_rwlock_wrlock(&fw->requests_lock);
    
    ml_async_request_t* req = fw->active_requests[request_id % fw->max_requests];
    if (req && req->request_id == request_id && req->status == ML_STATUS_PENDING) {
        req->status = ML_STATUS_CANCELLED;
        fw->active_requests[request_id % fw->max_requests] = NULL;
        
        /* TODO: Cancel ONNX RunAsync if possible */
        
        if (req->callback) {
            req->callback(req->user_data, NULL, 0, ML_STATUS_CANCELLED);
        }
        
        /* Cleanup */
        if (req->input_tensor) fw->ort_api->ReleaseValue(req->input_tensor);
        if (req->run_options) fw->ort_api->ReleaseRunOptions(req->run_options);
        ml_free_input_buffer(fw, req->input_buffer);
        ml_free_output_buffer(fw, req->output_buffer);
        free(req);
        
        pthread_rwlock_unlock(&fw->requests_lock);
        return 0;
    }
    
    pthread_rwlock_unlock(&fw->requests_lock);
    return -1;
}

ml_status_t ml_get_request_status(ml_framework_t* fw, uint64_t request_id) {
    if (!fw) return ML_STATUS_ERROR;
    
    pthread_rwlock_rdlock(&fw->requests_lock);
    
    ml_async_request_t* req = fw->active_requests[request_id % fw->max_requests];
    ml_status_t status = ML_STATUS_ERROR;
    
    if (req && req->request_id == request_id) {
        status = req->status;
    }
    
    pthread_rwlock_unlock(&fw->requests_lock);
    return status;
}

/* Memory management */
void* ml_alloc_input_buffer(ml_framework_t* fw, size_t size) {
    if (!fw || size > fw->config.memory.input_buffer_size) return NULL;
    return pool_alloc(fw->input_pool);
}

void* ml_alloc_output_buffer(ml_framework_t* fw, size_t size) {
    if (!fw || size > fw->config.memory.output_buffer_size) return NULL;
    return pool_alloc(fw->output_pool);
}

void ml_free_input_buffer(ml_framework_t* fw, void* buffer) {
    if (fw && buffer) pool_free(fw->input_pool, buffer);
}

void ml_free_output_buffer(ml_framework_t* fw, void* buffer) {
    if (fw && buffer) pool_free(fw->output_pool, buffer);
}

/* PP Core functions */
int ml_register_pp_core(ml_framework_t* fw, uint32_t pp_core_id) {
    if (!fw || pp_core_id >= fw->num_channels) return -1;
    return 0;  /* Already created during init */
}

ml_async_request_t* ml_poll_response(ml_framework_t* fw, uint32_t pp_core_id) {
    if (!fw || pp_core_id >= fw->num_channels) return NULL;
    
    ml_core_channel_t* channel = &fw->channels[pp_core_id];
    return (ml_async_request_t*)ring_dequeue(channel->response_ring);
}

/* Enhanced statistics */
void ml_get_stats(ml_framework_t* fw, char* buffer, size_t size) {
    if (!fw || !buffer || size == 0) return;
    
    int offset = 0;
    offset += snprintf(buffer + offset, size - offset,
        "=== ML Framework Statistics ===\n"
        "Total Requests: %lu\n"
        "Completed: %lu\n"
        "Failed: %lu\n"
        "Timeouts: %lu\n"
        "Overall Success Rate: %.2f%%\n"
        "Average Latency: %.2f ms\n\n",
        fw->stats.total_requests,
        fw->stats.completed_requests,
        fw->stats.failed_requests,
        fw->stats.timeout_requests,
        fw->stats.total_requests > 0 ? 
            (fw->stats.completed_requests * 100.0 / fw->stats.total_requests) : 0,
        fw->stats.completed_requests > 0 ?
            (fw->stats.total_inference_time / 1000.0 / fw->stats.completed_requests) : 0
    );
    
    /* Per model type statistics */
    offset += snprintf(buffer + offset, size - offset, "Per Model Type Statistics:\n");
    
    for (int i = 0; i <= MODEL_TYPE_CUSTOM; i++) {
        if (fw->stats.requests_per_type[i] > 0) {
            offset += snprintf(buffer + offset, size - offset,
                "  %s: %lu requests, avg latency: %.2f ms\n",
                model_type_to_string(i),
                fw->stats.requests_per_type[i],
                fw->stats.latency_per_type[i] / 1000.0 / fw->stats.requests_per_type[i]
            );
        }
    }
    
    /* Individual model statistics */
    offset += snprintf(buffer + offset, size - offset, "\nIndividual Model Statistics:\n");
    
    pthread_rwlock_rdlock(&fw->registry->lock);
    for (size_t i = 0; i < fw->registry->count; i++) {
        ml_model_t* model = fw->registry->models[i];
        if (model && model->stats.total_inferences > 0) {
            offset += snprintf(buffer + offset, size - offset,
                "  %s (%s):\n"
                "    Total inferences: %lu\n"
                "    Avg latency: %.2f ms\n"
                "    P99 latency: %.2f ms\n",
                model->name, model_type_to_string(model->type),
                model->stats.total_inferences,
                model->stats.avg_latency_ms,
                model->stats.p99_latency_ms
            );
        }
    }
    pthread_rwlock_unlock(&fw->registry->lock);
}

/* Model introspection */
void ml_print_model_info(ml_model_t* model) {
    if (!model) return;
    
    printf("\n=== Model Information ===\n");
    printf("Name: %s\n", model->name);
    printf("Type: %s\n", model_type_to_string(model->type));
    printf("Description: %s\n", model->description ? model->description : "N/A");
    printf("Version: %s\n", model->version);
    printf("Path: %s\n", model->path);
    printf("Supports Batching: %s (max size: %zu)\n", 
           model->supports_batching ? "Yes" : "No", model->max_batch_size);
    
    printf("\nInputs (%zu):\n", model->num_inputs);
    for (size_t i = 0; i < model->num_inputs; i++) {
        printf("  [%zu] %s: ", i, model->inputs[i].name);
        for (size_t j = 0; j < model->inputs[i].num_dimensions; j++) {
            printf("%lld%s", model->inputs[i].shape[j], 
                   j < model->inputs[i].num_dimensions - 1 ? "x" : "");
        }
        printf(" (%s)\n", model->inputs[i].is_dynamic ? "dynamic" : "static");
    }
    
    printf("\nOutputs (%zu):\n", model->num_outputs);
    for (size_t i = 0; i < model->num_outputs; i++) {
        printf("  [%zu] %s\n", i, model->outputs[i].name);
    }
    
    printf("\nPreprocessing: %s\n", 
           model->preprocess.type == PREPROCESS_NONE ? "None" :
           model->preprocess.type == PREPROCESS_STANDARDIZE ? "Standardization" :
           model->preprocess.type == PREPROCESS_MINMAX ? "Min-Max Scaling" : "Custom");
    
    printf("\nPerformance Hints:\n");
    printf("  Prefer GPU: %s\n", model->perf_hints.prefer_gpu ? "Yes" : "No");
    printf("  Threads: %d\n", model->perf_hints.num_threads);
    printf("  Max Memory: %zu MB\n", model->perf_hints.max_memory_mb);
}
