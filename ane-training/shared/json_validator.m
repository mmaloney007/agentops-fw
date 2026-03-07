#import <Foundation/Foundation.h>
#include "json_validator.h"

NSDictionary* extract_json(NSString *text) {
    if (!text) return nil;

    NSData *data = [text dataUsingEncoding:NSUTF8StringEncoding];
    id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if ([parsed isKindOfClass:[NSDictionary class]]) return parsed;

    NSRange start = [text rangeOfString:@"```json"];
    if (start.location != NSNotFound) {
        NSRange end = [text rangeOfString:@"```" options:0
                                    range:NSMakeRange(start.location + start.length,
                                                      text.length - start.location - start.length)];
        if (end.location != NSNotFound) {
            NSString *block = [text substringWithRange:
                NSMakeRange(start.location + start.length,
                            end.location - start.location - start.length)];
            data = [block dataUsingEncoding:NSUTF8StringEncoding];
            parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
            if ([parsed isKindOfClass:[NSDictionary class]]) return parsed;
        }
    }

    NSRange open = [text rangeOfString:@"{"];
    NSRange close = [text rangeOfString:@"}" options:NSBackwardsSearch];
    if (open.location != NSNotFound && close.location != NSNotFound && close.location > open.location) {
        NSString *block = [text substringWithRange:
            NSMakeRange(open.location, close.location - open.location + 1)];
        data = [block dataUsingEncoding:NSUTF8StringEncoding];
        parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if ([parsed isKindOfClass:[NSDictionary class]]) return parsed;
    }

    return nil;
}

float validate_fields(NSDictionary *parsed, NSDictionary *schema) {
    if (!parsed || !schema) return 0.0f;

    NSArray *required = schema[@"required"];
    if (!required || required.count == 0) {
        NSDictionary *properties = schema[@"properties"];
        if (properties) {
            required = properties.allKeys;
        } else {
            return parsed.count > 0 ? 1.0f : 0.0f;
        }
    }

    int present = 0;
    for (NSString *field in required) {
        if (parsed[field] != nil && parsed[field] != [NSNull null]) {
            present++;
        }
    }
    return (float)present / (float)required.count;
}

float composite_reward(NSString *response, NSDictionary *schema) {
    NSDictionary *parsed = extract_json(response);
    float json_valid = (parsed != nil) ? 1.0f : 0.0f;
    float field_score = validate_fields(parsed, schema);
    return 0.7f * field_score + 0.3f * json_valid;
}
