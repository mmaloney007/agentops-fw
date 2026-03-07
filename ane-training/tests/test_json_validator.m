#import <Foundation/Foundation.h>
#include "../shared/json_validator.h"
#include <assert.h>
#include <math.h>

int main(void) {
    @autoreleasepool {
        NSDictionary *r = extract_json(@"{\"name\": \"Alice\", \"age\": 30}");
        assert(r != nil);
        assert([r[@"name"] isEqualToString:@"Alice"]);

        r = extract_json(@"Here is the JSON:\n```json\n{\"x\": 1}\n```\nDone.");
        assert(r != nil);
        assert([r[@"x"] intValue] == 1);

        r = extract_json(@"The answer is {\"result\": true} ok");
        assert(r != nil);
        assert([r[@"result"] boolValue] == YES);

        assert(extract_json(@"no json here") == nil);

        NSDictionary *schema = @{
            @"required": @[@"name", @"age", @"city"],
            @"properties": @{@"name": @{}, @"age": @{}, @"city": @{}}
        };
        NSDictionary *full = @{@"name": @"A", @"age": @30, @"city": @"X"};
        assert(fabsf(validate_fields(full, schema) - 1.0f) < 0.01f);

        NSDictionary *partial = @{@"name": @"A"};
        assert(fabsf(validate_fields(partial, schema) - 0.333f) < 0.01f);

        float reward = composite_reward(@"{\"name\":\"A\",\"age\":30,\"city\":\"X\"}", schema);
        assert(reward > 0.9f);

        float bad_reward = composite_reward(@"garbage", schema);
        assert(bad_reward < 0.01f);

        NSLog(@"PASS: json_validator");
    }
    return 0;
}
