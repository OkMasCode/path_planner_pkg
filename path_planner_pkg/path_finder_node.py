import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Path, OccupancyGrid, Odometry
import numpy as np
from path_planner_pkg.utils.a_star import AStar
from path_planner_pkg.utils.utils import get_map, Movements8Connectivity, map_to_world, get_map_data, world_to_map

class GlobalPlannerNode(Node):
    def __init__(self):
        """Initialize the ROS2 node."""
        super().__init__('global_planner_node')

        self.map_path = '/home/francesco-masin/ros2_ws/src/path_planner_pkg/path_planner_pkg/maps/map.yaml'
        self.xy_reso = 3 #down-sampling integer factor along each axis
        self.goal_origin = np.array([0.0, 0.0]) # Placeholder until rviz updates it
        self.start_origin = np.array([0.0, 0.0])  
        self.replan_frequency = 0.5
        self.origin3 = np.array([0,0,0])  #origin for the path (it is different from origin for the map)
        # Publishers for /global_path and /global_costmap
        self.path_publisher = self.create_publisher(Path, '/global_path', 10)
        self.costmap_publisher = self.create_publisher(OccupancyGrid, '/global_costmap', 10)

        #subcriber to get the goal
        self.goal_subscription = self.create_subscription(PointStamped,'/clicked_point',self.goal_callback,10)
        
        # Load the map and get metadata parameters
        _, self.grid_map, self.metadata = get_map(self.map_path, self.xy_reso)
        """
        self.resolution = self.metadata["resolution"]
        self.origin = self.metadata["origin"]
        self.width = self.metadata["width"]
        self.height = self.metadata["height"]
        """
        #I get the metadata for the original map in order to print it in rviz 
        #otherwise I would print the scaled map and I don't know how to scale back 
        self.npmap, self.metadata2 = get_map_data(self.map_path)
        self.resolution2 = self.metadata2["resolution"]
        self.origin2 = self.metadata2["origin"]
        self.width2 = self.metadata2["width"]
        self.height2 = self.metadata2["height"]

        # Planner setup
        self.planner = AStar(self.grid_map, Movements8Connectivity())

        # Schedule periodic replanning
        self.timer = self.create_timer(1.0 / self.replan_frequency, self.plan_and_publish)
        self.start = world_to_map(self.start_origin, self.resolution2*self.xy_reso, self.origin2[:2])
        self.goal = world_to_map(self.goal_origin, self.resolution2*self.xy_reso, self.origin2[:2])
    
    def goal_callback(self, msg : PointStamped):

        map_pos = np.array([msg.point.x, msg.point.y])
        self.goal = world_to_map(map_pos, self.resolution2*self.xy_reso, self.origin2[:2])

    def plan_and_publish(self):
        """Plan the global path and publish the path and costmap."""

        self.get_logger().info(f'Planning global path from start {self.start} to goal {self.goal}')
        # Plan the path
        path = self.planner.plan(self.start, self.goal)
        world_path = []
        #rescale the path
        for map_pos in path:
            world_pos = map_to_world(map_pos, self.resolution2*self.xy_reso, self.origin2)
            world_path.append(world_pos)
        world_path = np.array(world_path)
        if world_path is not None:
            self.publish_path(world_path)
            self.publish_costmap()

    def publish_path(self, path):
        """Convert the planned path to a nav_msgs/Path and publish it."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'

        for point in path:
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'odom'
            pose.pose.position.x = float(point[0])  # x = col * resolution
            pose.pose.position.y = float(point[1])  # y = row * resolution
            pose.pose.position.z = 0.0
            path_msg.poses.append(pose)

        self.path_publisher.publish(path_msg)
        self.get_logger().info('Published global path')

    def publish_costmap(self):
        """Convert the planner's costmap to a nav_msgs/OccupancyGrid and publish it."""

        inverted_grid_map = self.npmap
        costmap = inverted_grid_map.copy() + 1
        # add a gaussian near the obstacles to avoid them
        for i in range(costmap.shape[0]):
            for j in range(costmap.shape[1]):
                if inverted_grid_map[i, j] > 0:
                    costmap[i - 2 : i + 3, j - 2 : j + 3] += 10
                    costmap[i - 1 : i + 2, j - 1 : j + 2] += 20
        costmap[inverted_grid_map > 0] = 255
        costmap[costmap > 255] = 255

        grid_msg = OccupancyGrid()
        
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'odom'

        grid_msg.info.resolution = self.resolution2  # Resolution of the grid
        grid_msg.info.width = self.width2
        grid_msg.info.height = self.height2
        grid_msg.info.origin.position.x = self.origin2[0]
        grid_msg.info.origin.position.y = self.origin2[1]
        grid_msg.info.origin.position.z = self.origin2[2]
        grid_msg.info.origin.orientation.w = 1.0
        # Flatten costmap and convert to list for OccupancyGrid (values 0-100)
        scaled_costmap = (costmap/255)*100
        flattened_costmap = scaled_costmap.flatten().astype(np.int8).tolist()
        grid_msg.data = flattened_costmap

        self.costmap_publisher.publish(grid_msg)
        self.get_logger().info('Published global costmap')

def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
