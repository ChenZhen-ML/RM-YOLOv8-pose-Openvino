#include "detector.h"

static constexpr int INPUT_W = 640;    // Width of input
static constexpr int INPUT_H = 640;    // Height of input

static constexpr int NUM_CLASSES = 18;  // Number of classes

static constexpr float BBOX_CONF_THRESH = 0.4;
static constexpr float NMS_THRESH = 0.4;

static constexpr float MERGE_CONF_ERROR = 0.15;
static constexpr float MERGE_MIN_IOU = 0.8;

static constexpr int TOPK = 128;       // TopK

ArmorDetector::ArmorDetector(){}

ArmorDetector::~ArmorDetector(){}

const vector<string> coconame = { "B_G","B1","B2","B3","B4","B5","B_o","B_Bs","B_Bb",
                                  "R_G","R1","R2","R3","R4","R5","R_o","R_Bs","R_Bb"};

/**
 * @brief find the max element in the array
 * @param ptr the pointer of the array
 * @param len the length of the array
 * @return max element in the array
 */
static inline int element_max(const float *ptr, int len)
{
    int max_arg = 0;
    for (int i = 1; i < len; i++) {
        if (ptr[i] > ptr[max_arg]) max_arg = i;
    }
    return max_arg;
}

/**
 * @brief Resize the image using "letterbox"
 * @param img Image before resize
 * @param transform_matrix Transform Matrix of Resize
 * @return Image after resize
 */
Mat ArmorDetector::preprocess_img(cv::Mat& img)
{
    float r = std::min(INPUT_W / (img.cols * 1.0), INPUT_H / (img.rows * 1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;

    int dw = INPUT_W - unpad_w; //填充
    int dh = INPUT_H - unpad_h; //

//    dw /= 2;
//    dh /= 2;

    Mat re;
    cv::resize(img, re, Size(unpad_w,unpad_h),0,0,cv::INTER_AREA); //按比例缩放
    Mat out;
    cv::copyMakeBorder(re, out, 0, dh, 0, dw, BORDER_CONSTANT,Scalar(114,114,114)); //填充

    this->rx = (float)img.cols / (float)(out.cols - dw);
    this->ry = (float)img.rows / (float)(out.rows - dh);

    return out;
}


/**
 * @brief Generate grids and stride.
 * @param target_w Width of input.
 * @param target_h Height of input.
 * @param strides A vector of stride.
 * @param grid_strides Grid stride generated in this function.
 */


void minMaxLoc(const float *scores,double &score,int &id)
{
    id = 0;
    score = scores[0];
    for(int i=0;i<18;i++)
    {
        if(scores[i]>scores[id])
        {
            id = i;
            score = scores[i];
        }
    }
}





void ArmorDetector::generateYoloxProposals(Mat& frame, const float* feat_ptr,float prob_threshold)
{
    std::vector<cv::Rect> boxes;
    vector<int> class_ids;
    vector<float> confidences;
    //Travel all the anchors
    for (int j = 0; j < 8400; j++) {
        const float classes_scores[18] = {
                feat_ptr[4 * 8400 + j],
                feat_ptr[5 * 8400 + j],
                feat_ptr[6 * 8400 + j],
                feat_ptr[7 * 8400 + j],
                feat_ptr[8 * 8400 + j],
                feat_ptr[9 * 8400 + j],
                feat_ptr[10 * 8400 + j],
                feat_ptr[11 * 8400 + j],
                feat_ptr[12 * 8400 + j],
                feat_ptr[13 * 8400 + j],
                feat_ptr[14 * 8400 + j],
                feat_ptr[15 * 8400 + j],
                feat_ptr[16 * 8400 + j],
                feat_ptr[17 * 8400 + j],
                feat_ptr[18 * 8400 + j],
                feat_ptr[19 * 8400 + j],
                feat_ptr[20 * 8400 + j],
                feat_ptr[21 * 8400 + j],

        };

        double score;
        int id;

        minMaxLoc(classes_scores, score, id);

        if (score > 0.3) {
            //cout<<score<<endl;
            const float cx = feat_ptr[0 * 8400 + j];
            const float cy = feat_ptr[1 * 8400 + j];
            const float ow = feat_ptr[2 * 8400 + j];
            const float oh = feat_ptr[3 * 8400 + j];

            cv::Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));
            box.y = static_cast<int>((cy - 0.5 * oh));
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);

            cout << score << endl;
            boxes.push_back(box);
            class_ids.push_back(id);
            confidences.push_back(score);
        }
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, BBOX_CONF_THRESH, NMS_THRESH, nms_result);

    std::vector<Detection> output;
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
    cout<<"output_size:"<<output.size()<< endl;

    for (int i = 0; i < output.size(); i++)
    {
        auto detection = output[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        // if (classId != 0) continue;
        auto confidence = detection.confidence;

        box.x = this->rx * box.x;
        box.y = this->ry * box.y;
        box.width = this->rx * box.width;
        box.height = this->ry * box.height;


        float xmax = box.x + box.width;
        float ymax = box.y + box.height;

        // detection box

        cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(xmax, ymax), Scalar(255,0,0), 3);

        // Detection box text
        std::string classString = coconame[classId] + ' ' + std::to_string(confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
        cv::rectangle(frame, textBox, Scalar(255,0,0), cv::FILLED);
        cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

        // cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(xmax, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
        // cv::putText(frame, coconame[classId], cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cv::imshow("frame",frame);
    waitKey(10);
}



//初始化
bool ArmorDetector::initModel(string xml_path,double cof_threshold,double nms_area_threshold)
{
    _xml_path = xml_path;
    //_bin_path = bin_path;
    _cof_threshold = cof_threshold;
    _nms_area_threshold = nms_area_threshold;
    Core ie;

    // Step 1. Read a model in OpenVINO Intermediate Representation (.xml and
    // .bin files) or ONNX (.onnx file) format\


    auto cnnNetwork = ie.ReadNetwork(_xml_path);

    //auto cnnNetwork = ie.ReadNetwork(_xml_path,_bin_path);
    // Step 2. Configure input & output
    // 输入设置
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;
    input->setPrecision(Precision::FP32);

    input->getInputData()->setLayout(Layout::NCHW);

    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();

    SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);

    //输出设置
    _output_name=cnnNetwork.getOutputsInfo().begin()->first;
    _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());

//    SizeVector size={8400,30};
    _outputinfo.begin()->second->setPrecision(Precision::FP32);
    // _outputinfo.begin()->second->reshape(size,Layout::ANY);

    // Step 3. Loading a model to the device
    // executable_network = ie.LoadNetwork(network, "MULTI:GPU");
    //_network =  ie.LoadNetwork(cnnNetwork, "GPU");
    _network =  ie.LoadNetwork(cnnNetwork, "CPU");
    return true;
}


//处理图像获取结果
bool ArmorDetector::process_frame(Mat& src)
{
    if(src.empty()){
        cout << "无效图片输入" << endl;
        return false;
    }
    Mat inframe = preprocess_img(src);

    ////////////////
    cv::Mat pre;
    cv::Mat pre_split[3];
    inframe.convertTo(pre,CV_32F);
    cv::split(pre/255.0f,pre_split);
    ////////////////////


    size_t img_size = 640*640;
    InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();

    Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();

    auto img_offset = INPUT_W * INPUT_H;
    //Copy img into blob
    for(int c = 2;c>=0;c--)
    {
        memcpy(blob_data, pre_split[c].data, INPUT_W * INPUT_H * sizeof(float));
        blob_data += img_offset;
    }


    //执行预测
    auto start = chrono::high_resolution_clock::now();
    infer_request->Infer();
    auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    cout<<"use "<<diff.count()<<" s" << endl;


    Blob::Ptr output_Blob = infer_request->GetBlob(_output_name);

    moutput = as<MemoryBlob>(output_Blob);
    auto moutputHolder = moutput->rmap();

    const float* net_pred = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type*>();

    generateYoloxProposals(src, net_pred, BBOX_CONF_THRESH);
}

