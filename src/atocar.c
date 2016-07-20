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
#define labelNum 1
//char *label_names[] = {"pedestrian"};
char *label_names[] = {"person", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "aeroplane", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
image atocar_labels[labelNum];


extern char* training_file;
void train_atocar(char *cfgfile, char *weightfile)
{
	char *base = basecfg(cfgfile);
	printf("%s\n", base);
	network net = parse_network_cfg(cfgfile);
    char *train_images = training_file;
	char *backup_directory = backup;
    data_seed = time(0);
    float avg_loss = -1;
    
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
    char **paths = (char **)list_to_array(plist);//?б

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
	FILE* log;
	if(log=fopen(logFile,"w+")){
		printf("Log File Path:%s\n",logFile);
	}
	else{
		printf("%s open failed!\n",logFile);
	}
	setlinebuf(log);
	time=clock();
	int current_batch=0;
    while( (current_batch=get_current_batch(net)) < net.max_batches){
        i += 1;
		
        
        pthread_join(load_thread, 0);
        
		train = buffer;
        load_thread = load_data_in_thread(args);
		
        //printf("Loaded: %lf seconds\n", sec(clock()-time));
		//printf("in0\n");
        
		//showImg();
		//printf("test img_path:%s\n",img_path);
		
		//testDetection(net);
        
		float loss = train_network(net, train);
		
        //printf("in1\n");
		if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
		//printf("in2\n");
		printf("total use = %.0lf seconds, left time = %.0lf seconds\n",sec(clock()-time),sec(clock()-time)/current_batch*(net.max_batches-current_batch));
        printf("time %d: loss %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0){
            char buff[256];
            sprintf(buff, "%s/atocar_%d.weights", backup_directory,  i);
            save_weights(net, buff);
        }
		fprintf(log,"total use = %.0lf seconds, left time = %.0lf seconds\n",sec(clock()-time),sec(clock()-time)/current_batch*(net.max_batches-current_batch));
		fprintf(log,"time=%d loss=%f\n",i,loss);
		//printf("in3\n");
        free_data(train);
    }
	fclose(log);
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
			//printf("hah i=%d num =%d,n=%d\n",i,num,n);
            int index = i*num + n;
			//printf("index=%d\n",index);
            int p_index = side*side*classes + i*num + n;
			//printf("classes=%d\n",classes);
            float scale = predictions[p_index];
			//printf("scale=%f\n",scale);
            //printf("index=%d i=%d n=%d classes=%d ",index,i,n,classes);
			int box_index = side*side*(classes + num) + (i*num + n)*4;
            //printf("index=%d i=%d n=%d classes=%d ",index,i,n,classes);
			boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
            //printf("in\n");
			//printf("i=%d n=%d classes=%d ",i,n,classes);
			for(j = 0; j < classes; ++j){
				//printf("i=%d j=%d n=%d classes=%d ",i,j,n,classes);
				//printf("in1\n");
                int class_index = i*classes;
                //printf("in2\n");
				float prob = scale*predictions[class_index+j];
				//printf("in3\n");
				//printf("probs=%f\n",probs[index][j]);
				
				//printf("in4 index = %d j=%d\n",index,j);
                probs[index][j] = (prob > thresh) ? prob : 0;
				
				//printf("in4 index = %d j=%d\n",index,j);
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
			//printf("out\n");
			
			
        }
		
		
    }
}


void genOutputImgPath(char* filename,char outputFileImgName[],int start){
	int i;
	for(i = strlen(filename)-1 ; i>=0 ;i-- ){
		if(filename[i]=='/'){
				break;
		}
	}
	int j;
	for(j = 0 ; j < strlen(filename)-i-1 ; j++){
		outputFileImgName[start+j]=filename[i+j+1];
	}
	outputFileImgName[start+j]='\0';
}

void test_atocar(char *cfgfile, char *weightfile, char *filename, float thresh)
{
	
    network net = parse_network_cfg(cfgfile);//??

	if(weightfile){
        load_weights(&net, weightfile);//weights?
    }

    detection_layer l = net.layers[net.n-1];

	set_batch_network(&net, 1);
 
	srand(2222222);//TODO should be real random
    clock_t time;
	
    char buff[256];
	
    char *input = buff;
    int j;
	
    float nms=.5;
	
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));//float x, y, w, h;?boundingbox
    
	float **probs = calloc(l.side*l.side*l.n, sizeof(float *));//?boundingboxиfloat飬??    
	for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    
	
	FILE *file = fopen(test_file, "r");
    if(!file){
		printf("test file load failed!\n");
	}
	else{
		printf("test list is %s\n",test_file);
	}
    
	char outputFileImgName[512];
	int i;
	//printf("strlen(testOutputDir) = %d\n",strlen(testOutputDir));
	for(i = 0 ; i < strlen(testOutputDir); i++){
		outputFileImgName[i] = testOutputDir[i];
	}
	int start=strlen(testOutputDir);
	if(outputFileImgName[start-1]!='/'){
		outputFileImgName[start]='/';
		start++;
	}
	
	while( (filename=fgetl(file)) ){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
		printf("start to test %s\n",filename);
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        
		float *predictions = network_predict(net, X);
        //printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        
		
		convert_detections_atocar(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        
		if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, label_names, atocar_labels, 20);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, label_names, atocar_labels, labelNum);
        
		genOutputImgPath(filename,outputFileImgName,start);
		//printf("%s\n",outputFileImgName);
		save_image(im, outputFileImgName);
        //show_image(im, "predictions");

        //show_image(sized, "resized");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
    }
	fclose(file);
}

void draw(){
	image im = load_image("/home/intern/Desktop/dataset/INRIAPerson/train/images/crop001001.png",0,0,0);
	draw_bbx(im,0.471882640587 ,0.417008196721, 0.305623471883 ,0.610655737705);
	draw_bbx(im,0.146699266504 ,0.531762295082 ,0.21760391198 ,0.395491803279);
	draw_bbx(im,0.267726161369 ,0.420081967213 ,0.173594132029, 0.473360655738);
	save_image(im, "predictions");
}

void test(){
	//_mkdir("../testNow");
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

    float thresh = find_float_arg(argc, argv, "-thresh", .2);//??0.2,С0.2???boundingbox
    int cam_index = find_int_arg(argc, argv, "-c", 0);//unknow
    int frame_skip = find_int_arg(argc, argv, "-s", 0);//unknow
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;//weights
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_atocar(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "train")) train_atocar(cfg, weights);
	else if(0==strcmp(argv[2], "draw")) draw();
    //else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
    //else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, label_names, atocar_labels, 20, frame_skip);
}
