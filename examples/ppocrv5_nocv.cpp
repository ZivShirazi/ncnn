// pip install paddlepaddle==3.0.0
// pip install paddleocr==3.0.0
// paddlex --install paddle2onnx
// paddleocr ocr -i test.png
// paddlex --paddle2onnx --paddle_model_dir ~/.paddlex/official_models/PP-OCRv5_mobile_det --onnx_model_dir PP-OCRv5_mobile_det
// paddlex --paddle2onnx --paddle_model_dir ~/.paddlex/official_models/PP-OCRv5_mobile_rec --onnx_model_dir PP-OCRv5_mobile_rec
// pnnx PP-OCRv5_mobile_det.onnx inputshape=[1,3,320,320] inputshape2=[1,3,256,256]
// pnnx PP-OCRv5_mobile_rec.onnx inputshape=[1,3,48,160] inputshape2=[1,3,48,256]
// pnnx PP-OCRv5_server_det.onnx inputshape=[1,3,320,320] inputshape2=[1,3,256,256] fp16=0
// pnnx PP-OCRv5_server_rec.onnx inputshape=[1,3,48,160] inputshape2=[1,3,48,256] fp16=0
// convert to buffer ussing ncnn2mem

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "layer.h"
#include "mat.h"
#include "net.h"

#include "PP_OCRv5_mobile_det.mem.h"
#include "PP_OCRv5_mobile_rec_opt_int8.mem.h"
#include "PP_OCRv5_mobile_det.id.h"
#include "PP_OCRv5_mobile_rec_opt_int8.id.h"
#include "PP_OCRv5_mobile_rec.mem.h"
#include "PP_OCRv5_mobile_rec.id.h"

#include "ppocrv5_dict.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <vector>

// One decoded token from the recognizer output sequence.
struct Character
{
    int id;
    float prob;
};

// Text region description in image space.
// (cx, cy): region center
// (w, h): box size used for recognition crop (w = short side, h = long side)
// u/v: orthonormal axes of oriented rectangle
struct Object
{
    float cx;
    float cy;
    float w;
    float h;
    float ux;
    float uy;
    float vx;
    float vy;
    float angle;
    int orientation;
    float prob;
    std::vector<Character> text;
};

// CTC greedy decoding (plain-language version):
//
// The recognizer does NOT output a final string directly.
// Instead, for each time step (possible recognized character) along the text line, it outputs a probability
// distribution over all possible symbols:
//
//   out.h = number of time steps
//   out.w = number of classes (class 0 is the special "blank" token)
//
// CTC (Connectionist Temporal Classification) is a common OCR sequence format.
// In CTC, the network can emit:
// - a real character class (1..N), or
// - blank (0), which means "no character at this step".
//
// This function applies standard greedy CTC decoding:
// 1) At each time step, choose the class with maximum probability (argmax).
// 2) If it is the same as the previous chosen class, skip it
//    (collapse repeated runs like A A A -> A).
// 3) If the chosen class is blank (0), skip it.
// 4) Otherwise, store the decoded symbol id (index - 1) and its probability.
//
// The resulting token list is later mapped to UTF-8 strings by character_dict.
static int decode_ctc_text(const ncnn::Mat& out, std::vector<Character>& text)
{
    if (out.empty())
        return -1;

    int last_token = 0;

    for (int i = 0; i < out.h; i++)
    {
        const float* p = out.row(i);

        int index = 0;
        float max_score = -9999.f;
        for (int j = 0; j < out.w; j++)
        {
            float score = *p++;
            if (score > max_score)
            {
                max_score = score;
                index = j;
            }
        }

        // CTC rule, if index is same as last one, they will be merged into one token
        if (last_token == index)
            continue;

        last_token = index;

        if (index <= 0)
            continue;

        Character ch;
        ch.id = index - 1;
        ch.prob = max_score;
        text.push_back(ch);
    }

    return 0;
}

// Crop text region into recognizer input canvas (target_h=48) via ncnn affine warp.
static int get_rotate_crop_image(const unsigned char* rgb, int img_w, int img_h, const Object& object, std::vector<unsigned char>& crop_rgb, int& crop_w, int& crop_h)
{
    // The recognizer model is trained with 48 pixel height and variable width (depending on the aspect ratio of the text line).
    const int target_height = 48;
    const float rw = std::max(object.w, 1.f);
    const float rh = std::max(object.h, 1.f);
    const int target_width = std::max(32, std::min(640, (int)roundf(rh * target_height / rw)));

    crop_w = target_width;
    crop_h = target_height;

    if (crop_w <= 1 || crop_h <= 1)
        return -1;

    std::vector<unsigned char> affine_rgb((size_t)crop_w * crop_h * 3);

    // Build source rectangle corners from object center/axes.
    const float hh = object.h * 0.5f;
    const float hw = object.w * 0.5f;

    // Caculating the 3 corners of the source rectangle. The 4th corner can be inferred since it's a parallelogram.
    const float p0x = object.cx - object.ux * hh - object.vx * hw;
    const float p0y = object.cy - object.uy * hh - object.vy * hw;
    const float p1x = object.cx + object.ux * hh - object.vx * hw;
    const float p1y = object.cy + object.uy * hh - object.vy * hw;
    const float p2x = object.cx - object.ux * hh + object.vx * hw;
    const float p2y = object.cy - object.uy * hh + object.vy * hw;

    // Calculate a 2x3 affine transform matrix that represents the
    // rotation and scaling from the source rectangle to the destination rectangle.
    float src_pts[6] = {p0x, p0y, p1x, p1y, p2x, p2y};
    float dst_pts[6] = {0.f, 0.f, (float)crop_w, 0.f, 0.f, (float)crop_h};
    float tm[6];
    ncnn::get_affine_transform(dst_pts, src_pts, 3, tm);

    // Effectively do: affine_rgb = warp_affine(rgb, tm), with bilinear sampling and border replication.
    // Which means that we rotate the text region to be horizontal, and resize it to the target height and variable width.
    ncnn::warpaffine_bilinear_c3(rgb, img_w, img_h, affine_rgb.data(), crop_w, crop_h, tm);

    // Get the text in the right direction according to the principal axis. This is important for recognizer performance.
    if (object.ux < 0.f)
    {
        // Keep text direction stable when principal axis points to the left.
        std::vector<unsigned char> rotated_rgb((size_t)crop_w * crop_h * 3);
        ncnn::kanna_rotate_c3(affine_rgb.data(), crop_w, crop_h, rotated_rgb.data(), crop_w, crop_h, 3);
        crop_rgb.swap(rotated_rgb);
    }
    else
    {
        crop_rgb.swap(affine_rgb);
    }

    return 0;
}

class PPOCRv5
{
public:
    void init();
    void deinit();

    void detect(const unsigned char* rgb, int img_w, int img_h, std::vector<Object>& objects);

    void recognize(const unsigned char* rgb, int img_w, int img_h, Object& object);

protected:
    ncnn::Net ppocrv5_det;
    ncnn::Net ppocrv5_rec;
};

void PPOCRv5::deinit()
{
    ppocrv5_det.clear();
    ppocrv5_rec.clear();
}

void PPOCRv5::init()
{
    ppocrv5_det.opt.use_vulkan_compute = false;
#if NCNN_VULKAN
    ppocrv5_det.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN
    ppocrv5_det.load_param(PP_OCRv5_mobile_det_ncnn_param_bin);
    ppocrv5_det.load_model(PP_OCRv5_mobile_det_ncnn_bin);

    ppocrv5_rec.opt.use_vulkan_compute = false;
#if NCNN_VULKAN
    ppocrv5_rec.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN
    ppocrv5_rec.load_param(PP_OCRv5_mobile_rec_opt_int8_ncnn_param_bin);
    ppocrv5_rec.load_model(PP_OCRv5_mobile_rec_opt_int8_ncnn_bin);
}

void PPOCRv5::detect(const unsigned char* rgb, int img_w, int img_h, std::vector<Object>& objects)
{
    // Keep detector input behavior aligned with ppocrv5.cpp:
    // resize + stride letterbox + identical normalization.
    const int target_size = 960;
    const int target_stride = 32;

    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (std::max(w, h) > target_size)
    {
        if (w > h)
        {
            scale = (float)target_size / w;
            w = target_size;
            h = (int)(h * scale);
        }
        else
        {
            scale = (float)target_size / h;
            h = target_size;
            w = (int)(w * scale);
        }
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);

    // Calculate how much pixels to add for each side to make the padded size divisible by target_stride.
    int wpad = (w + target_stride - 1) / target_stride * target_stride - w;
    int hpad = (h + target_stride - 1) / target_stride * target_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    // Those preprocessing values are specific for the model used in this example, and may need to be changed for different models.
    // Look for substract_mean_normalize in the docs for further explanation.
    const float mean_vals[3] = {0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f};
    const float norm_vals[3] = {1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f};
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = ppocrv5_det.create_extractor();

    ex.input(PP_OCRv5_mobile_det_ncnn_param_id::LAYER_in0, in_pad);

    ncnn::Mat out;
    ex.extract(PP_OCRv5_mobile_det_ncnn_param_id::BLOB_out0, out);

    // Detector out0 is probability map in [0,1]. We convert it to [0,255].
    // The dimensions of out are the same as the padded input,
    // so we can map detected boxes back to original image space via the scale factor and padding size.
    const float denorm_vals[1] = {255.f};
    out.substract_mean_normalize(0, denorm_vals);

    // Any pixel with a probability higher than this is considered "text."
    const float threshold = 0.3f;
    // The box confidence threshold. After finding a box,
    // the average probability of all pixels inside it must be at least 0.6 for the box to be kept.
    const float box_thresh = 0.6f;
    // Used to expand the detected text region slightly.
    // Since the model is trained to detect the "shrunk" kernel of the text,
    // this ratio helps restore the box to its original full size.
    const float enlarge_ratio = 1.95f;
    const float min_size = 3 * scale;
    // Ignores tiny boxes (likely noise) that are smaller than a few pixels.
    const int max_candidates = 1000;

    const int map_w = out.w;
    const int map_h = out.h;

    // Convert probability map to pixel map for easier processing.
    // Use PIXEL_GRAY since we only have one channel, and we want the output to be in [0,255].
    std::vector<unsigned char> pred((size_t)map_w * map_h, 0);
    out.to_pixels(pred.data(), ncnn::Mat::PIXEL_GRAY);

    // Threshold map to bitmap, then run connected components to get candidates.
    std::vector<unsigned char> bitmap((size_t)map_w * map_h, 0);
    for (int y = 0; y < map_h; y++)
    {
        const unsigned char* p = pred.data() + (size_t)y * map_w;
        unsigned char* b = bitmap.data() + (size_t)y * map_w;
        for (int x = 0; x < map_w; x++)
        {
            b[x] = p[x] > threshold * 255.f ? 1 : 0;
        }
    }

    std::vector<int> visited((size_t)map_w * map_h, 0);
    std::vector<int> q;
    q.reserve((size_t)map_w * map_h / 4);

    int candidates = 0;
    for (int y = 0; y < map_h && candidates < max_candidates; y++)
    {
        for (int x = 0; x < map_w && candidates < max_candidates; x++)
        {
            size_t idx = (size_t)y * map_w + x;
            if (!bitmap[idx] || visited[idx])
                continue;

            candidates++;
            q.clear();
            q.push_back((int)idx);
            visited[idx] = 1;

            int qhead = 0;
            int minx = x;
            int miny = y;
            int maxx = x;
            int maxy = y;
            float score_sum = 0.f;
            int score_count = 0;

            while (qhead < (int)q.size())
            {
                int cur = q[qhead++];
                int cy = cur / map_w;
                int cx = cur - cy * map_w;

                minx = std::min(minx, cx);
                miny = std::min(miny, cy);
                maxx = std::max(maxx, cx);
                maxy = std::max(maxy, cy);

                score_sum += pred[(size_t)cy * map_w + cx];
                score_count++;

                const int nx4[4] = {cx - 1, cx + 1, cx, cx};
                const int ny4[4] = {cy, cy, cy - 1, cy + 1};
                for (int k = 0; k < 4; k++)
                {
                    int nx = nx4[k];
                    int ny = ny4[k];
                    if (nx < 0 || nx >= map_w || ny < 0 || ny >= map_h)
                        continue;

                    size_t nidx = (size_t)ny * map_w + nx;
                    if (!bitmap[nidx] || visited[nidx])
                        continue;

                    visited[nidx] = 1;
                    q.push_back((int)nidx);
                }
            }

            if (score_count == 0)
                continue;

            // Check that the average score of the connected component is above box_thresh,
            //  and its size is above min_size. If not, skip it as a false positive.
            float score = score_sum / score_count / 255.f;
            if (score < box_thresh)
                continue;

            float aabb_w = (float)(maxx - minx + 1);
            float aabb_h = (float)(maxy - miny + 1);
            float maxwh = std::max(aabb_w, aabb_h);
            if (maxwh < min_size)
                continue;

            const int n = score_count;

            // Everything below is for rotated box fitting.
            // We want to find the minimum-area rotated rectangle that encloses all pixels in the connected component.
            // If OpenCV is available, we can just call minAreaRect. But since this example is for no-OpenCV build,
            // we implement a simple PCA-based rotated rectangle fitting here.
            // If we only need axis-aligned boxes, we can just use the AABB (minx, miny, maxx, maxy) without all this.

            // Compute principal direction (PCA) from foreground pixels.
            // This approximates minAreaRect orientation without OpenCV.
            double sumx = 0.0;
            double sumy = 0.0;
            for (int i = 0; i < n; i++)
            {
                int cur = q[i];
                int py = cur / map_w;
                int px = cur - py * map_w;
                sumx += px;
                sumy += py;
            }

            float mx = (float)(sumx / n);
            float my = (float)(sumy / n);

            double cxx = 0.0;
            double cyy = 0.0;
            double cxy = 0.0;
            for (int i = 0; i < n; i++)
            {
                int cur = q[i];
                int py = cur / map_w;
                int px = cur - py * map_w;
                double dx = px - mx;
                double dy = py - my;
                cxx += dx * dx;
                cyy += dy * dy;
                cxy += dx * dy;
            }

            float theta = 0.f;
            if (n > 1)
                theta = 0.5f * atan2f((float)(2.0 * cxy), (float)(cxx - cyy));

            float ux = cosf(theta);
            float uy = sinf(theta);
            float vx = -uy;
            float vy = ux;

            // Project component pixels to local axes to get oriented extents.
            float min_u = 1e20f;
            float max_u = -1e20f;
            float min_v = 1e20f;
            float max_v = -1e20f;

            for (int i = 0; i < n; i++)
            {
                int cur = q[i];
                int py = cur / map_w;
                int px = cur - py * map_w;

                float dx = px - mx;
                float dy = py - my;

                float pu = dx * ux + dy * uy;
                float pv = dx * vx + dy * vy;

                min_u = std::min(min_u, pu);
                max_u = std::max(max_u, pu);
                min_v = std::min(min_v, pv);
                max_v = std::max(max_v, pv);
            }

            float major = max_u - min_u + 1.f;
            float minor = max_v - min_v + 1.f;

            float center_u = (min_u + max_u) * 0.5f;
            float center_v = (min_v + max_v) * 0.5f;

            float cx = mx + ux * center_u + vx * center_v;
            float cy = my + uy * center_u + vy * center_v;

            if (major < minor)
            {
                // Ensure major >= minor so text long side is consistently h.
                std::swap(major, minor);
                std::swap(ux, vx);
                std::swap(uy, vy);
            }

            float rw = minor;
            float rh = major;

            rw *= enlarge_ratio;
            rh += minor * (enlarge_ratio - 1.f);

            // Undo letterbox/padding back to original image coordinates.
            cx = (cx - (wpad / 2.f)) / scale;
            cy = (cy - (hpad / 2.f)) / scale;
            rw = rw / scale;
            rh = rh / scale;

            if (rw <= 1.f || rh <= 1.f)
                continue;

            int orientation = 0;
            float angle = atan2f(uy, ux) * 180.f / 3.14159265f;
            if (angle < 0.f)
                angle += 180.f;

            Object obj;
            obj.cx = cx;
            obj.cy = cy;
            obj.w = rw;
            obj.h = rh;
            obj.ux = ux;
            obj.uy = uy;
            obj.vx = vx;
            obj.vy = vy;
            obj.angle = angle;
            obj.orientation = orientation;
            obj.prob = score;
            objects.push_back(obj);
        }
    }

    std::sort(objects.begin(), objects.end(), [](const Object& a, const Object& b) {
        if (fabsf(a.cy - b.cy) > 10.f)
            return a.cy < b.cy;
        return a.cx < b.cx;
    });
}

void PPOCRv5::recognize(const unsigned char* rgb, int img_w, int img_h, Object& object)
{
    // Crop oriented line image, normalize, then run recognizer.
    std::vector<unsigned char> crop_rgb;
    int crop_w = 0;
    int crop_h = 0;
    if (get_rotate_crop_image(rgb, img_w, img_h, object, crop_rgb, crop_w, crop_h) != 0)
        return;

    // Create a matrix and convert from RGB to BGR
    ncnn::Mat in = ncnn::Mat::from_pixels(crop_rgb.data(), ncnn::Mat::PIXEL_RGB2BGR, crop_w, crop_h);

    //Scale the data from 0 to 255 integers into -1.0 to 1.0 floating-point numbers.
    // Those preprocessing values are specific for the model used in this example, and may need to be changed for different models.
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
    const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = ppocrv5_rec.create_extractor();
    ex.input(PP_OCRv5_mobile_rec_opt_int8_ncnn_param_id::LAYER_in0, in);

    ncnn::Mat out;
    ex.extract(PP_OCRv5_mobile_rec_opt_int8_ncnn_param_id::BLOB_out0, out);

    decode_ctc_text(out, object.text);
}

static int detect_ppocrv5(const unsigned char* rgb, int img_w, int img_h, std::vector<Object>& objects)
{
    // Same high-level flow as ppocrv5.cpp: init -> detect -> recognize per object.
    PPOCRv5 ppocrv5;

    ppocrv5.init();

    ppocrv5.detect(rgb, img_w, img_h, objects);

    for (size_t i = 0; i < objects.size(); i++)
    {
        ppocrv5.recognize(rgb, img_w, img_h, objects[i]);
    }

    ppocrv5.deinit();

    return 0;
}

static int draw_objects(const std::vector<Object>& objects)
{
    // Console output format intentionally mirrors ppocrv5.cpp.
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%s %.5f at %.2f %.2f %.2f x %.2f  @ %.2f  =  ", obj.orientation == 0 ? "H" : "V", obj.prob,
                obj.cx, obj.cy, obj.w, obj.h, obj.angle);

        std::string text;
        for (size_t j = 0; j < objects[i].text.size(); j++)
        {
            const Character& ch = objects[i].text[j];
            if (ch.id >= character_dict_size)
                continue;

            text += character_dict[ch.id];
        }
        fprintf(stderr, "%s\n", text.c_str());
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    int img_w = 0;
    int img_h = 0;
    int img_c = 0;

    unsigned char* rgb = stbi_load(imagepath, &img_w, &img_h, &img_c, 3);
    if (!rgb)
    {
        fprintf(stderr, "stbi_load %s failed\n", imagepath);
        return -1;
    }

    std::vector<Object> objects;
    detect_ppocrv5(rgb, img_w, img_h, objects);

    // Convert indexes to charcters using the character_dict, then print the results.
    draw_objects(objects);

    stbi_image_free(rgb);

    return 0;
}
