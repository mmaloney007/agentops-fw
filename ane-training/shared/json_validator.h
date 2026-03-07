#ifndef JSON_VALIDATOR_H
#define JSON_VALIDATOR_H

#import <Foundation/Foundation.h>

NSDictionary* extract_json(NSString *text);
float validate_fields(NSDictionary *parsed, NSDictionary *schema);
float composite_reward(NSString *response, NSDictionary *schema);

#endif
