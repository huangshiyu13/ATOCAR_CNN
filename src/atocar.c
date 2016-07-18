#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"

#include "time.h"
#include "stdio.h"
#include "stdlib.h"

#include "data.h"
#include "image.h"
#include "cuda.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif
#define labelNum 20
//char *label_names[] = {"pedestrian"};
char *label_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
image atocar_labels[labelNum];

void testDetection(network net){
	float thresh = 0.5;
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);//TODO should be real random
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));//float x, y, w, h;ôboundingbox
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));//ÿboundingboxиfloat飬Ըֵ
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
  
        
        image im = load_image_color(img_path,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        
		float *predictions = network_predict(net, X);
        //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        convert_detections_atocar(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, label_names, atocar_labels, 20);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, label_names, atocar_labels, labelNum);
        
		save_image(im, "predictions");
        show_image(im, "predictions");

        show_image(sized, "resized");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
	//printf("%f %f %f %f %f\n",boxes[0].x,boxes[0].y,boxes[0].w,boxes[0].h,probs[0]);
	//getchar();
}

void train_atocar(char *cfgfile, char *weightfile)
{
    //char *train_images = "C:/Users/huangsy13/Desktop/test/data/VOCtest_06-Nov-2007/2007_test.txt";//TODO read from configure file
    char *train_images = "../data/train.txt";
	char *backup_directory = "backup/";
    //srand(time(0));//
    data_seed = time(0);
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    //printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;//unknow

    list *plist = get_paths(train_images);//image·
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);//תб

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
		
        time=clock();
        pthread_join(load_thread, 0);
        
		train = buffer;
        load_thread = load_data_in_thread(args);
		
        //printf("Loaded: %lf seconds\n", sec(clock()-time));
		//printf("in0\n");
        time=clock();
		//showImg();
		//printf("test img_path:%s\n",img_path);
		
		//testDetection(net);
        
		float loss = train_network(net, train);
        //printf("in1\n");
		if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
		//printf("in2\n");
        if(i%20 == 0) printf("time %d: loss %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        //if(i%1000==0 || (i < 1000 && i%100 == 0)){
        //    char buff[256];
        //    sprintf(buff, "%s/atocar_%d.weights", backup_directory,  i);
        //    save_weights(net, buff);
        //}
		//printf("in3\n");
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/atocar.weights", backup_directory);
    save_weights(net, buff);
}

void convert_detections_atocar(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    //int per_cell = 5*num+classes;
    for (i = 0; i < side*side; ++i){
        int row = i / side;
        int col = i % side;
        for(n = 0; n < num; ++n){
            int index = i*num + n;
            int p_index = side*side*classes + i*num + n;
            float scale = predictions[p_index];
            int box_index = side*side*(classes + num) + (i*num + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
            for(j = 0; j < classes; ++j){
                int class_index = i*classes;
                float prob = scale*predictions[class_index+j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}

void print_atocar_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
										 xmin, ymin, xmax, ymax);
        }
    }
}

void validate_atocar(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("/home/pjreddie/data/voc/2007_test.txt");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, label_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 2;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            convert_detections_atocar(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
            if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);
            print_atocar_detections(fps, id, boxes, probs, side*side*l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void test_atocar(char *cfgfile, char *weightfile, char *filename, float thresh)//ԵͼƬ
{

    network net = parse_network_cfg(cfgfile);//ļظ
    if(weightfile){
        load_weights(&net, weightfile);//weightsļ
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);//TODO should be real random
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));//float x, y, w, h;ôboundingbox
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));//ÿboundingboxиfloat飬Ըֵ
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        
		float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        convert_detections_atocar(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, label_names, atocar_labels, 20);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, label_names, atocar_labels, labelNum);
        
		save_image(im, "predictions");
        show_image(im, "predictions");

        show_image(sized, "resized");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}

void test(){
		
} 

void run_atocar(int argc, char **argv)
{
	test();
	//return;
    int i;
    for(i = 0; i < labelNum; ++i){
        char buff[256];
        sprintf(buff, "data/labels/%s.png", label_names[i]);
        atocar_labels[i] = load_image_color(buff, 0, 0);
    }

    float thresh = find_float_arg(argc, argv, "-thresh", .2);//ĬΪ0.2,С0.2Ĳչʾboundingbox
    int cam_index = find_int_arg(argc, argv, "-c", 0);//unknow
    int frame_skip = find_int_arg(argc, argv, "-s", 0);//unknow
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];//ļ
    char *weights = (argc > 4) ? argv[4] : 0;//weightsļ
    char *filename = (argc > 5) ? argv[5]: 0;//ôļ
    if(0==strcmp(argv[2], "test")) test_atocar(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "train")) train_atocar(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_atocar(cfg, weights);
    //else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
    //else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, label_names, atocar_labels, 20, frame_skip);
}
