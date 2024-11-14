#include "nodes/osd/vp_osd_node.h"
#include "nodes/track/vp_sort_track_node.h"
#include "nodes/vp_file_src_node.h"
#include "nodes/vp_screen_des_node.h"
#include "nodes/vp_split_node.h"
#include "utils/analysis_board/vp_analysis_board.h"
#include "vp_trtyolo_detector.h"

int main() {
    // Disable logging code location and thread ID
    VP_SET_LOG_INCLUDE_CODE_LOCATION(false);
    VP_SET_LOG_INCLUDE_THREAD_ID(false);
    VP_LOGGER_INIT();

    // Video sources
    auto file_src_0 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_0", 0, "demo0.mp4");
    auto file_src_1 = std::make_shared<vp_nodes::vp_file_src_node>("file_src_1", 1, "demo1.mp4");

    // Inference node (TensorRT-YOLO detector)
    auto detector = std::make_shared<vp_nodes::vp_trtyolo_detector>("yolo_detector", "yolo11n.engine", "labels.txt", true, 2);

    // Tracking node (SORT tracker)
    auto track = std::make_shared<vp_nodes::vp_sort_track_node>("track");

    // OSD (On-Screen Display) node
    auto osd = std::make_shared<vp_nodes::vp_osd_node>("osd");

    // Channel splitting node
    auto split = std::make_shared<vp_nodes::vp_split_node>("split_by_channel", true);

    // Local display nodes
    auto screen_des_0 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_0", 0);
    auto screen_des_1 = std::make_shared<vp_nodes::vp_screen_des_node>("screen_des_1", 1);

    // Constructing the pipeline
    detector->attach_to({file_src_0, file_src_1});
    track->attach_to({detector});
    osd->attach_to({track});
    split->attach_to({osd});

    // Splitting by vp_split_node for display
    screen_des_0->attach_to({split});
    screen_des_1->attach_to({split});

    // Start video sources
    file_src_0->start();
    file_src_1->start();

    // Debugging: Display analysis board
    vp_utils::vp_analysis_board board({file_src_0, file_src_1});
    board.display(1, false);  // Display board with refresh rate of 1 second, non-verbose

    // Wait for user input to stop and detach nodes recursively
    std::string wait;
    std::getline(std::cin, wait);
    file_src_0->detach_recursively();
    file_src_1->detach_recursively();
}
