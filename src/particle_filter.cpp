/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 20;  // TODO: Set the number of particles

  // Assigning std deviations for the later Gaussian noise add operation
  double std_x, std_y, std_theta;
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];

  // Creating Gaussian distribution for x, y and theta with acc. std deviation for adding noise
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // Random engine for picking random values from distribution
  default_random_engine gen;

  // Initializing num_particles particles acc. generated distributions with weight 1.0
  for (int i = 0; i < num_particles; i++)
  {
      // Create particle P and initialize
      Particle P;
      P.id = i;
      P.x = dist_x(gen);
      P.y = dist_y(gen);
      P.theta = dist_theta(gen);
      P.weight = 1.0;

      // Append particle P and weights to respective vectors
      particles.push_back(P);
      weights.push_back(P.weight);
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

   // Assigning std deviations for the later Gaussian noise add operation
   double std_x, std_y, std_theta;
   std_x = std_pos[0];
   std_y = std_pos[1];
   std_theta = std_pos[2];

   // Random engine for picking random values from distribution
   default_random_engine gen;

   for (int i = 0; i < num_particles; i++)
   {
       // Assign current particle location values
       double p_current_x, p_current_y, p_current_theta;
       p_current_x = particles[i].x;
       p_current_y = particles[i].y;
       p_current_theta = particles[i].theta;

       // Declare predicted particle location and orientation values
       double p_pred_x, p_pred_y, p_pred_theta;

       // Predict particle position and orientation
       if (fabs(yaw_rate) > 0.001) // Distinguish for very small yaw_rate to avoid division by zero
       {
           p_pred_x = p_current_x + (velocity / yaw_rate) * \
           (sin(p_current_theta + yaw_rate * delta_t) - sin(p_current_theta));
           p_pred_y = p_current_y + (velocity / yaw_rate) * \
           (cos(p_current_theta) - cos(p_current_theta + yaw_rate * delta_t));
           p_pred_theta = p_current_theta + (yaw_rate * delta_t);
       }
       else
       {
           p_pred_x = p_current_x + (velocity * delta_t * cos(p_current_theta));
           p_pred_y = p_current_y + (velocity * delta_t * sin(p_current_theta));
           p_pred_theta = p_current_theta;
       }

       // Creating Gaussian distribution for x, y and theta with acc. std deviation for adding noise
       normal_distribution<double> dist_x(p_pred_x, std_x);
       normal_distribution<double> dist_y(p_pred_y, std_y);
       normal_distribution<double> dist_theta(p_pred_theta, std_theta);

       // Assign predicted location and orientation back to particle
       particles[i].x = dist_x(gen);
       particles[i].y = dist_y(gen);
       particles[i].theta = dist_theta(gen);
   }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */

  // Iterate over all observations
  for (unsigned int i = 0; i < observations.size(); i++)
  {
      // Define closest value to initially maximum distance for minimum likelihood
      double closest_dist = numeric_limits<double>::max();
      // Define initial landmark id for closest landmark
      int closest_id = -1;

      // Iterated over all predictions
      for (unsigned int j = 0; j < predicted.size(); j++)
      {
          // Calculate distance between observed and predicted landmark
          double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

          // Determine if predicted is at the closest distance to obersations landmark
          if (distance < closest_dist)
          {
              closest_dist = distance;
              closest_id = predicted[j].id;
          }

      }

      observations[i].id = closest_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // Iterate over all particles
  for (int i = 0; i < num_particles; i++)
  {
      // Assign particle location and orientation
      double p_x, p_y, p_theta;
      p_x = particles[i].x;
      p_y = particles[i].y;
      p_theta = particles[i].theta;

      // Create vector to hold landmark positions predicted to be within sensor range
      vector<LandmarkObs> LandmarkObs_InSensorRange;

      // Iterate over all landmarks in map
      for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
      {
          // Assign landmark position and id
          double landmark_pos_x, landmark_pos_y;
          int landmark_id;
          landmark_pos_x = map_landmarks.landmark_list[j].x_f;
          landmark_pos_y = map_landmarks.landmark_list[j].y_f;
          landmark_id = map_landmarks.landmark_list[j].id_i;

          // Calculate distance of particle to landmark in map
          double distance = dist(landmark_pos_x, landmark_pos_y, p_x, p_y);

          if (distance <= sensor_range)
          {
              LandmarkObs_InSensorRange.push_back(LandmarkObs{landmark_id, landmark_pos_x, landmark_pos_y});
          }
      }

      // Create vector for transformed observations from car coordinates into map coordinates
      vector<LandmarkObs> obs_transf;
      // Iterate over all observations
      for (unsigned int k = 0; k < observations.size(); k++)
      {
          double obs_map_x, obs_map_y;
          // Transform observations in car coordinates to map coordinates
          obs_map_x = p_x + (cos(p_theta) * observations[k].x) - \
          (sin(p_theta) * observations[k].y);
          obs_map_y = p_y + (sin(p_theta) * observations[k].x) + \
          (cos(p_theta) * observations[k].y);

          obs_transf.push_back(LandmarkObs{observations[k].id, obs_map_x, obs_map_y});
      }

      // Call helper function dataAssociation
      dataAssociation(LandmarkObs_InSensorRange, obs_transf);

      // Re-initialize weights
      particles[i].weight = 1.0;

      // Iterate over all transformed observations in sensor range
      for (unsigned int l = 0; l < obs_transf.size(); l++)
      {
          double obs_map_x, obs_map_y;
          double mean_landmark_x, mean_landmark_y;
          double std_landmark_x, std_landmark_y;

          // Assign transformed observations
          obs_map_x = obs_transf[l].x;
          obs_map_y = obs_transf[l].y;

          // Iterate over all landmarks in sensor range
          for (unsigned int m = 0; m < LandmarkObs_InSensorRange.size(); m++)
          {
              // Identify map landmarks that correspond to transformed observed ones
              if (obs_transf[l].id == LandmarkObs_InSensorRange[m].id)
              {
                  mean_landmark_x = LandmarkObs_InSensorRange[m].x;
                  mean_landmark_y = LandmarkObs_InSensorRange[m].y;
                  break;
              }
          }

          std_landmark_x = std_landmark[0];
          std_landmark_y = std_landmark[1];
          // Define variable for weight derived from multivariate normal distribution
          double obs_weight;
          obs_weight = (1 / (2 * M_PI * std_landmark_x * std_landmark_y)) * \
          exp(-(pow(mean_landmark_x - obs_map_x, 2) / (2 * pow(std_landmark_x, 2)) + \
          (pow(mean_landmark_y - obs_map_y, 2) / (2 * pow(std_landmark_y, 2)))));

          particles[i].weight *= obs_weight;
      }
  }

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  vector<Particle> new_particles; // new particle set from re-sampling

  vector<double> current_weights;
  // Iterate over all particles
  for (int i = 0; i < num_particles; i++)
  {
      current_weights.push_back(particles[i].weight);
  }

  // Random engine for picking random values from distribution
  default_random_engine gen;

  // Random start index for re-sampling wheel
  uniform_int_distribution<int> uniform_particle_dist(0, num_particles-1);
  int index = uniform_particle_dist(gen);

  // Determine maximum weight
  double max_current_weight = *max_element(current_weights.begin(), current_weights.end());

  // Uniform random distributions to pick random step size for re-sampling
  uniform_real_distribution<double> uniform_weight_dist(0.0, max_current_weight);

  double temp = 0.0;

  // Start re-sampling
  for (int j = 0; j < num_particles; j++)
  {
      temp += uniform_weight_dist(gen) * 2.0;
      while (temp > current_weights[index])
      {
          temp -= current_weights[index];
          index = (index + 1) % num_particles; // cyclical re-sampling
      }

      new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
