/**
 * Billing: the credit ledger, Stripe checkout, Stripe's signed messages, refunds and the admin tools.
 * Stripe itself is a stand-in here: no request leaves this computer.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import crypto from 'node:crypto';
import mongoose from 'mongoose';
import TestStartup from '../utils/test.startup.js';
import ApiClient from '../utils/api.client.js';
import { Server } from '../../server/server.js';
import User from '../../server/models/user.model.js';
import CreditLine, { BILLING_SERVICE, Checkout } from '../../server/models/billing.model.js';
import * as billing from '../../server/controllers/billing.controller.js';

describe('Billing', () => {
    let testStartup;
    let client;
    let buyer;
    let other;
    const run = crypto.randomBytes(4).toString('hex');
    const stripe = { calls: [], refuse: null, count: 0, paid: new Set(), sessions: new Map() };

    const standIn = async (url, options) => {
        const path = new URL(url).pathname.replace('/v1', '');
        const fields = Object.fromEntries(new URLSearchParams(options.body));
        stripe.calls.push({ path, fields, headers: options.headers });
        if (stripe.refuse) return Response.json({ error: { message: stripe.refuse } }, { status: 402 });
        stripe.count += 1;
        if (options.method === 'GET' && path === '/checkout/sessions') {
            const page = stripe.sessions.get(new URL(url).searchParams.get('payment_intent'));
            return Response.json({ object: 'list', data: page ? [page] : [] });
        }
        if (path === '/checkout/sessions') {
            return Response.json({ id: `cs_test_${run}_${stripe.count}`, url: `https://checkout.stripe.com/c/pay/cs_test_${run}_${stripe.count}` });
        }
        const expiring = path.match(/^\/checkout\/sessions\/(.+)\/expire$/);
        if (expiring) {
            if (stripe.paid.has(expiring[1])) return Response.json({ error: { message: 'This Checkout Session is already complete.' } }, { status: 400 });
            return Response.json({ id: expiring[1], status: 'expired' });
        }
        if (path === '/refunds') return Response.json({ id: `re_${run}_${stripe.count}`, status: 'succeeded' });
        return Response.json({ error: { message: 'Unknown path' } }, { status: 404 });
    };

    const signature = (body, { secret = process.env.STRIPE_WEBHOOK_SECRET, time = Math.floor(Date.now() / 1000) } = {}) =>
        `t=${time},v1=${crypto.createHmac('sha256', secret).update(`${time}.${body}`).digest('hex')}`;

    const fromStripe = (event, options = {}) => {
        const body = JSON.stringify(event);
        return fetch(`${testStartup.baseURL}/api/v1/billing/stripe`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json; charset=utf-8', 'Stripe-Signature': options.header || signature(options.signedBody || body, options) },
            body
        });
    };

    const paid = (userId, cents, name, extra = {}) => ({
        id: `evt_${run}_${name}`,
        type: 'checkout.session.completed',
        data: {
            object: {
                id: `cs_test_${run}_${name}`,
                payment_status: 'paid',
                currency: 'usd',
                amount_subtotal: cents,
                amount_total: cents,
                payment_intent: `pi_${run}_${name}`,
                metadata: { service: BILLING_SERVICE, user: userId },
                ...extra
            }
        }
    });

    const refunded = (name, cents, eventName) => ({
        id: `evt_${run}_${eventName}`,
        type: 'charge.refunded',
        data: { object: { id: `ch_${run}_${name}`, payment_intent: `pi_${run}_${name}`, amount_refunded: cents } }
    });

    const balance = async (userId) => (await billing.balanceOf(userId)).balance;

    const makeOlder = (userId, days) =>
        User.collection.updateOne({ _id: new mongoose.Types.ObjectId(userId) }, { $set: { createdAt: new Date(Date.now() - days * 86400000) } });

    beforeAll(async () => {
        process.env.STRIPE_SECRET_KEY ||= 'sk_test_standin';
        process.env.STRIPE_WEBHOOK_SECRET ||= 'whsec_standin';
        testStartup = new TestStartup('billing');
        await testStartup.initialize();
        client = testStartup.getClient();
        billing.setStripeFetch(standIn);
        buyer = testStartup.user;
        other = testStartup.creator;
    }, 60000);

    afterAll(async () => {
        billing.setStripeFetch(null);
        await testStartup.cleanup();
    }, 30000);

    describe('Off until keys are set', () => {
        it('says it is off, and every other billing address is missing', async () => {
            const kept = { key: process.env.STRIPE_SECRET_KEY, secret: process.env.STRIPE_WEBHOOK_SECRET };
            delete process.env.STRIPE_SECRET_KEY;
            delete process.env.STRIPE_WEBHOOK_SECRET;
            try {
                await testStartup.loginAsUser('user');
                const state = await client.get('/api/v1/billing');
                expect(state.status).toBe(200);
                expect(state.data).toEqual({ success: true, on: false });

                const checkout = await client.post('/api/v1/billing/checkout', { amount: 1000 });
                expect(checkout.status).toBe(404);
                expect(checkout.data.code).toBe('BILLING_OFF');
                expect((await client.get('/api/v1/billing/lines')).status).toBe(404);

                const message = await fromStripe(paid(buyer.id, 1000, 'off'), { secret: kept.secret });
                expect(message.status).toBe(404);
                expect(await balance(buyer.id)).toBe(0);
            } finally {
                process.env.STRIPE_SECRET_KEY = kept.key;
                process.env.STRIPE_WEBHOOK_SECRET = kept.secret;
            }
        });

        it('refuses to start with one Stripe setting and not the other', () => {
            const kept = process.env.STRIPE_WEBHOOK_SECRET;
            delete process.env.STRIPE_WEBHOOK_SECRET;
            try {
                expect(() => Server.prototype.validateEnvironment.call({})).toThrow(/STRIPE_SECRET_KEY and STRIPE_WEBHOOK_SECRET/);
            } finally {
                process.env.STRIPE_WEBHOOK_SECRET = kept;
            }
            expect(() => Server.prototype.validateEnvironment.call({})).not.toThrow();
        });

        it('refuses a live Stripe key anywhere but production', () => {
            const kept = process.env.STRIPE_SECRET_KEY;
            try {
                for (const key of ['sk_live_notreal', 'rk_live_notreal']) {
                    process.env.STRIPE_SECRET_KEY = key;
                    expect(() => Server.prototype.validateEnvironment.call({})).toThrow(/only for production/);
                }
                process.env.STRIPE_SECRET_KEY = 'rk_test_notreal';
                expect(() => Server.prototype.validateEnvironment.call({})).not.toThrow();
            } finally {
                process.env.STRIPE_SECRET_KEY = kept;
            }
        });

        it('needs a sign-in for everything but Stripe\'s messages', async () => {
            const stranger = new ApiClient(testStartup.baseURL);
            expect((await stranger.get('/api/v1/billing')).status).toBe(401);
            expect((await stranger.get('/api/v1/billing/lines')).status).toBe(401);
        });
    });

    describe('Checkout', () => {
        it('shows a new account its balance and what it may buy', async () => {
            await testStartup.loginAsUser('user');
            const state = await client.get('/api/v1/billing');
            expect(state.status).toBe(200);
            expect(state.data).toMatchObject({ on: true, balance: 0, held: 0, currency: 'usd', smallest: 1000, most: 2500, newAccount: true, emailVerified: true });
        });

        it('refuses an amount that is too small, too large or not whole cents', async () => {
            await testStartup.loginAsUser('user');
            const before = stripe.calls.length;
            const small = await client.post('/api/v1/billing/checkout', { amount: 999 });
            expect(small.status).toBe(400);
            expect(small.data.code).toBe('TOO_SMALL');
            const large = await client.post('/api/v1/billing/checkout', { amount: 2501 });
            expect(large.status).toBe(400);
            expect(large.data).toMatchObject({ code: 'TOO_LARGE', most: 2500 });
            for (const amount of [10.5, '1000abc', -1000, 0, null]) {
                const bad = await client.post('/api/v1/billing/checkout', { amount });
                expect(bad.status).toBe(400);
                expect(bad.data.code).toBe('BAD_AMOUNT');
            }
            expect(stripe.calls.length).toBe(before);
        });

        it('sends a new account to Stripe\'s page with 3D Secure asked for, and adds no credit itself', async () => {
            await testStartup.loginAsUser('user');
            const response = await client.post('/api/v1/billing/checkout', { amount: 1000 });
            expect(response.status).toBe(200);
            expect(response.data.url).toMatch(/^https:\/\/checkout\.stripe\.com\//);

            const call = stripe.calls.at(-1);
            expect(call.path).toBe('/checkout/sessions');
            expect(call.headers.Authorization).toBe(`Bearer ${process.env.STRIPE_SECRET_KEY}`);
            expect(call.fields).toMatchObject({
                mode: 'payment',
                client_reference_id: buyer.id,
                'allowed_payment_method_types[0]': 'card',
                'line_items[0][quantity]': '1',
                'line_items[0][price_data][currency]': 'usd',
                'line_items[0][price_data][unit_amount]': '1000',
                'metadata[service]': BILLING_SERVICE,
                'metadata[user]': buyer.id,
                'metadata[project]': process.env.APP_NAME.toLowerCase(),
                'payment_intent_data[metadata][user]': buyer.id,
                'payment_intent_data[metadata][service]': BILLING_SERVICE,
                'payment_method_options[card][request_three_d_secure]': 'any'
            });
            expect(call.fields.success_url).toBe(`${process.env.APP_URL}/billing?paid=yes`);
            expect(call.fields['payment_intent_data[statement_descriptor_suffix]']).toMatch(/^[A-Z0-9 ]{1,10}$/);
            expect(call.fields['line_items[0][price_data][product_data][name]']).toBe(`${process.env.APP_NAME} credit`);
            expect(JSON.stringify(response.data)).not.toContain(process.env.STRIPE_SECRET_KEY);
            expect(await balance(buyer.id)).toBe(0);
        });

        it('lets an older account buy more, without forcing 3D Secure', async () => {
            await makeOlder(other.id, 30);
            await testStartup.loginAsUser('creator');
            const state = await client.get('/api/v1/billing');
            expect(state.data).toMatchObject({ newAccount: false, most: 50000 });
            const response = await client.post('/api/v1/billing/checkout', { amount: 20000 });
            expect(response.status).toBe(200);
            expect(stripe.calls.at(-1).fields['payment_method_options[card][request_three_d_secure]']).toBeUndefined();
            expect((await client.post('/api/v1/billing/checkout', { amount: 50001 })).data.code).toBe('TOO_LARGE');
        });

        it('refuses an account whose email is not verified', async () => {
            await User.collection.updateOne({ _id: new mongoose.Types.ObjectId(testStartup.superCreator.id) }, { $set: { emailVerified: false } });
            await testStartup.loginAsUser('superCreator');
            const response = await client.post('/api/v1/billing/checkout', { amount: 1000 });
            expect(response.status).toBe(403);
            expect(response.data.code).toBe('EMAIL_NOT_VERIFIED');
        });

        it('passes on Stripe\'s refusal without adding credit', async () => {
            await testStartup.loginAsUser('user');
            await Checkout.deleteMany({ user: buyer.id });
            stripe.refuse = 'Your account cannot currently make live charges.';
            try {
                const response = await client.post('/api/v1/billing/checkout', { amount: 1000 });
                expect(response.status).toBe(502);
                expect(response.data.code).toBe('STRIPE_REFUSED');
            } finally {
                stripe.refuse = null;
            }
            expect(await balance(buyer.id)).toBe(0);
        });
    });

    describe('Stripe\'s signed messages', () => {
        it('adds the credit when a paid message arrives, and tells the person', async () => {
            const response = await fromStripe(paid(buyer.id, 2000, 'first'));
            expect(response.status).toBe(200);
            expect(await balance(buyer.id)).toBe(2000);

            await testStartup.loginAsUser('user');
            const lines = await client.get('/api/v1/billing/lines');
            expect(lines.data.lines[0]).toMatchObject({ kind: 'topup', change: 2000, balance: 2000, paid: true });
            await expect.poll(async () => (await client.get('/api/v1/notifications')).data.notifications.some((entry) => entry.type === 'billing:credit'), { timeout: 5000 }).toBe(true);
        });

        it('counts a message once, however many times it arrives', async () => {
            const event = paid(buyer.id, 2000, 'first');
            const again = await Promise.all([fromStripe(event), fromStripe(event), fromStripe({ ...event, id: `evt_${run}_resent` })]);
            again.forEach((response) => expect(response.status).toBe(200));
            expect(await balance(buyer.id)).toBe(2000);
            expect(await CreditLine.countDocuments({ service: BILLING_SERVICE, user: buyer.id, kind: 'topup' })).toBe(1);
        });

        it('refuses a forged message', async () => {
            const event = paid(buyer.id, 5000, 'forged');
            const wrongSecret = await fromStripe(event, { secret: 'whsec_someone_else' });
            expect(wrongSecret.status).toBe(400);
            const noSignature = await fromStripe(event, { header: 't=1,v1=' });
            expect(noSignature.status).toBe(400);
            const nonsense = await fromStripe(event, { header: 'hello' });
            expect(nonsense.status).toBe(400);
            expect(await balance(buyer.id)).toBe(2000);
        });

        it('refuses a message altered after it was signed', async () => {
            const signedFor = JSON.stringify(paid(buyer.id, 1000, 'altered'));
            const response = await fromStripe(paid(buyer.id, 900000, 'altered'), { signedBody: signedFor });
            expect(response.status).toBe(400);
            expect(await balance(buyer.id)).toBe(2000);
        });

        it('refuses an old message sent again', async () => {
            const response = await fromStripe(paid(buyer.id, 1000, 'old'), { time: Math.floor(Date.now() / 1000) - 3600 });
            expect(response.status).toBe(400);
            expect(await balance(buyer.id)).toBe(2000);
        });

        it('ignores a payment that is not paid, is another app\'s, or is in another currency', async () => {
            // In a Stripe account shared with other work, their payments and refunds arrive here too and carry no mark.
            const cases = [
                paid(buyer.id, 1000, 'unmarked', { metadata: {} }),
                paid(buyer.id, 1000, 'no-metadata', { metadata: undefined }),
                paid(buyer.id, 1000, 'project-only', { metadata: { project: process.env.APP_NAME.toLowerCase(), user: buyer.id } }),
                refunded('somebody-elses-invoice', 5000, 'their-refund'),
                paid(buyer.id, 1000, 'unpaid', { payment_status: 'unpaid' }),
                paid(buyer.id, 1000, 'elsewhere', { metadata: { service: 'another-app', user: buyer.id } }),
                paid(buyer.id, 1000, 'euros', { currency: 'eur' }),
                paid('not-an-id', 1000, 'nobody'),
                { id: `evt_${run}_other`, type: 'customer.created', data: { object: {} } }
            ];
            for (const event of cases) expect((await fromStripe(event)).status).toBe(200);
            expect(await balance(buyer.id)).toBe(2000);
        });
    });

    describe('The ledger', () => {
        it('never lets the balance go below zero', async () => {
            await expect(billing.charge(buyer.id, 2001, { about: 'Too much' })).rejects.toMatchObject({ code: 'NOT_ENOUGH_CREDIT', statusCode: 402 });
            expect(await balance(buyer.id)).toBe(2000);
            await expect(billing.charge(buyer.id, 0)).rejects.toMatchObject({ code: 'BAD_AMOUNT' });
            await expect(billing.charge(buyer.id, 10.5)).rejects.toMatchObject({ code: 'BAD_AMOUNT' });
        });

        it('lets only one of two charges at the same moment spend the same credit', async () => {
            const results = await Promise.allSettled([
                billing.charge(buyer.id, 2000, { about: 'First' }),
                billing.charge(buyer.id, 2000, { about: 'Second' })
            ]);
            expect(results.filter((result) => result.status === 'fulfilled')).toHaveLength(1);
            expect(results.find((result) => result.status === 'rejected').reason.code).toBe('NOT_ENOUGH_CREDIT');
            expect(await balance(buyer.id)).toBe(0);
        });

        it('keeps every line when many are written at once', async () => {
            await billing.addCredit(buyer.id, 1000, { about: 'For the next test' });
            const results = await Promise.allSettled(Array.from({ length: 8 }, (unused, index) => billing.charge(buyer.id, 200, { about: `Charge ${index}` })));
            expect(results.filter((result) => result.status === 'fulfilled')).toHaveLength(5);
            expect(await balance(buyer.id)).toBe(0);

            const lines = await CreditLine.find({ service: BILLING_SERVICE, user: buyer.id }).sort({ seq: 1 });
            expect(lines.map((line) => line.seq)).toEqual(lines.map((unused, index) => index + 1));
            let running = 0;
            lines.forEach((line) => {
                running += line.amount;
                expect(line.balance).toBe(running);
                expect(line.balance).toBeGreaterThanOrEqual(0);
            });
        });

        it('counts a charge with the same key once', async () => {
            await billing.addCredit(buyer.id, 1000, { about: 'For the next test' });
            const first = await billing.charge(buyer.id, 300, { key: `job:${run}`, about: 'A job' });
            const second = await billing.charge(buyer.id, 300, { key: `job:${run}`, about: 'A job' });
            expect(first.repeated).toBe(false);
            expect(second.repeated).toBe(true);
            expect(await balance(buyer.id)).toBe(700);
        });

        it('sets credit aside, charges what was used and gives the rest back', async () => {
            await billing.setAside(buyer.id, 500, `rent:${run}`, 'Set aside');
            expect(await billing.balanceOf(buyer.id)).toEqual({ balance: 200, held: 500 });
            await expect(billing.charge(buyer.id, 201)).rejects.toMatchObject({ code: 'NOT_ENOUGH_CREDIT' });
            await expect(billing.setAside(buyer.id, 201, `rent2:${run}`)).rejects.toMatchObject({ code: 'NOT_ENOUGH_CREDIT' });

            const done = await billing.settle(buyer.id, `rent:${run}`, 120, 'Used 120');
            expect(done.line.toObject()).toMatchObject({ kind: 'charge', amount: 380, held: -500 });
            expect(await billing.balanceOf(buyer.id)).toEqual({ balance: 580, held: 0 });

            const twice = await billing.settle(buyer.id, `rent:${run}`, 500, 'Again');
            expect(twice.repeated).toBe(true);
            expect(await billing.balanceOf(buyer.id)).toEqual({ balance: 580, held: 0 });
            await expect(billing.settle(buyer.id, `never:${run}`, 1)).rejects.toMatchObject({ code: 'NOT_SET_ASIDE' });
        });

        it('charges no more than was set aside, and notes what was short', async () => {
            await billing.setAside(buyer.id, 100, `over:${run}`, 'Set aside');
            const done = await billing.settle(buyer.id, `over:${run}`, 130, 'Used more');
            expect(done.line.toObject()).toMatchObject({ amount: 0, held: -100, short: 30 });
            expect(await billing.balanceOf(buyer.id)).toEqual({ balance: 480, held: 0 });
        });

        it('has a balance that is the sum of its lines', async () => {
            const lines = await CreditLine.find({ service: BILLING_SERVICE, user: buyer.id });
            const totals = await billing.balanceOf(buyer.id);
            expect(lines.reduce((sum, line) => sum + line.amount, 0)).toBe(totals.balance);
            expect(lines.reduce((sum, line) => sum + line.held, 0)).toBe(totals.held);
        });

        it('never changes or removes a line', async () => {
            const line = await CreditLine.findOne({ service: BILLING_SERVICE, user: buyer.id });
            await expect(CreditLine.updateOne({ _id: line._id }, { amount: 999999 })).rejects.toThrow(/never changed/);
            await expect(CreditLine.deleteOne({ _id: line._id })).rejects.toThrow(/never changed/);
            await expect(CreditLine.findOneAndUpdate({ _id: line._id }, { balance: 999999 })).rejects.toThrow(/never changed/);
            line.amount = 999999;
            await line.save();
            expect((await CreditLine.findById(line._id)).amount).not.toBe(999999);
        });

        it('keeps each app\'s balance apart', async () => {
            await CreditLine.create({ service: 'another-app', user: buyer.id, live: false, seq: 1, kind: 'topup', amount: 77700, balance: 77700 });
            expect(await balance(buyer.id)).toBe(480);
            await testStartup.loginAsUser('user');
            const state = await client.get('/api/v1/billing');
            expect(state.data.balance).toBe(480);
            const lines = await client.get('/api/v1/billing/lines');
            expect(lines.data.lines.some((line) => line.change === 77700)).toBe(false);
        });

        it('shows a person only their own lines, in pages', async () => {
            await testStartup.loginAsUser('creator');
            const others = await client.get('/api/v1/billing/lines');
            expect(others.data.lines).toEqual([]);

            await testStartup.loginAsUser('user');
            const first = await client.get('/api/v1/billing/lines?limit=3');
            expect(first.data.lines).toHaveLength(3);
            expect(first.data.more).toBe(true);
            const next = await client.get(`/api/v1/billing/lines?limit=3&before=${first.data.lines.at(-1).seq}`);
            expect(next.data.lines[0].seq).toBe(first.data.lines.at(-1).seq - 1);
        });
    });

    describe('Refunds', () => {
        let payment;

        beforeAll(async () => {
            await fromStripe(paid(other.id, 3000, 'refundable'));
            payment = await CreditLine.findOne({ service: BILLING_SERVICE, user: other.id, kind: 'topup' });
        });

        it('is for administrators only', async () => {
            await testStartup.loginAsUser('creator');
            const response = await client.post(`/api/v1/billing/lines/${payment.id}/refund`, {});
            expect(response.status).toBe(403);
            expect(await balance(other.id)).toBe(3000);
        });

        it('takes credit back and asks Stripe to return the money', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.post(`/api/v1/billing/lines/${payment.id}/refund`, { amount: 1000, reason: 'Asked for it' });
            expect(response.status).toBe(201);
            expect(response.data).toMatchObject({ balance: 2000 });
            const call = stripe.calls.at(-1);
            expect(call.path).toBe('/refunds');
            expect(call.fields).toMatchObject({ payment_intent: `pi_${run}_refundable`, amount: '1000', 'metadata[service]': BILLING_SERVICE, 'metadata[project]': process.env.APP_NAME.toLowerCase() });
            expect(call.headers['Idempotency-Key']).toBeTruthy();
            expect(await balance(other.id)).toBe(2000);
        });

        it('does not take the same refund twice when Stripe reports it', async () => {
            expect((await fromStripe(refunded('refundable', 1000, 'refund-report'))).status).toBe(200);
            expect(await balance(other.id)).toBe(2000);
        });

        it('takes credit back once for a refund made at Stripe', async () => {
            const event = refunded('refundable', 1500, 'refund-at-stripe');
            expect((await fromStripe(event)).status).toBe(200);
            expect((await fromStripe(event)).status).toBe(200);
            expect(await balance(other.id)).toBe(1500);
        });

        it('refuses to refund more than was paid or more than is unspent', async () => {
            await testStartup.loginAsUser('admin');
            const tooMuch = await client.post(`/api/v1/billing/lines/${payment.id}/refund`, { amount: 1501 });
            expect(tooMuch.status).toBe(400);
            expect(tooMuch.data).toMatchObject({ code: 'MORE_THAN_PAID', left: 1500 });

            await billing.charge(other.id, 1000, { about: 'Spent' });
            const spent = await client.post(`/api/v1/billing/lines/${payment.id}/refund`, { amount: 1500 });
            expect(spent.status).toBe(402);
            expect(spent.data).toMatchObject({ code: 'NOT_ENOUGH_CREDIT', balance: 500 });
            expect(await balance(other.id)).toBe(500);
        });

        it('puts the credit back when Stripe does not make the refund', async () => {
            await testStartup.loginAsUser('admin');
            stripe.refuse = 'This charge has already been refunded.';
            try {
                const response = await client.post(`/api/v1/billing/lines/${payment.id}/refund`, { amount: 500 });
                expect(response.status).toBe(502);
            } finally {
                stripe.refuse = null;
            }
            expect(await balance(other.id)).toBe(500);
            const again = await client.post(`/api/v1/billing/lines/${payment.id}/refund`, {});
            expect(again.status).toBe(201);
            expect(await balance(other.id)).toBe(0);
        });

        it('takes what is left and notes the rest when a refund at Stripe is larger than the unspent credit', async () => {
            await fromStripe(paid(other.id, 1000, 'spent'));
            await billing.charge(other.id, 800, { about: 'Spent' });
            expect((await fromStripe(refunded('spent', 1000, 'refund-spent'))).status).toBe(200);
            expect(await balance(other.id)).toBe(0);
            const line = await CreditLine.findOne({ service: BILLING_SERVICE, user: other.id }).sort({ seq: -1 });
            expect(line.toObject()).toMatchObject({ kind: 'refund', amount: -200, short: 800 });
            expect((await fromStripe(refunded('spent', 1000, 'refund-spent-again'))).status).toBe(200);
            expect(await CreditLine.countDocuments({ service: BILLING_SERVICE, user: other.id, kind: 'refund', 'ref.payment': `pi_${run}_spent` })).toBe(1);
        });

        it('refuses to refund a line that is not a payment', async () => {
            await testStartup.loginAsUser('admin');
            const charge = await CreditLine.findOne({ service: BILLING_SERVICE, user: other.id, kind: 'charge' });
            expect((await client.post(`/api/v1/billing/lines/${charge.id}/refund`, {})).status).toBe(404);
            expect((await client.post('/api/v1/billing/lines/not-an-id/refund', {})).status).toBe(404);
        });
    });

    describe('One payment page at a time', () => {
        let person;
        let theirs;
        const buy = (cents) => theirs.post('/api/v1/billing/checkout', { amount: cents });
        const pages = () => Checkout.find({ service: BILLING_SERVICE, user: person.id });

        beforeAll(async () => {
            person = await testStartup.createMutableUser({ role: 'USER', prefix: `pages${run}` });
            await User.collection.updateOne({ _id: new mongoose.Types.ObjectId(person.id) }, { $set: { emailVerified: true } });
            theirs = new ApiClient(testStartup.baseURL);
            expect((await theirs.post('/api/v1/auth/login', person.credentials)).status).toBe(200);
        });

        it('closes the open page at Stripe before opening another, so a new account cannot pass its limit', async () => {
            expect((await buy(2500)).status).toBe(200);
            const [first] = await pages();
            expect(first).toMatchObject({ amount: 2500, live: false });
            expect(first.endsAt.getTime()).toBeGreaterThan(Date.now() + 30 * 60000);
            expect(stripe.calls.at(-1).fields.expires_at).toBe(String(Math.floor(first.endsAt.getTime() / 1000)));

            expect((await buy(1000)).status).toBe(200);
            expect(stripe.calls.at(-2).path).toBe(`/checkout/sessions/${first.session}/expire`);
            const now = await pages();
            expect(now.map((page) => page.amount)).toEqual([1000]);
        });

        it('opens one page when two are asked for at the same moment', async () => {
            const both = await Promise.all([buy(1000), buy(1000)]);
            expect(both.map((response) => response.status).sort()).toEqual([200, 409]);
            expect(both.find((response) => response.status === 409).data.code).toBe('PAYMENT_UNDER_WAY');
            expect(await pages()).toHaveLength(1);
        });

        it('waits for a page that was paid meanwhile, then counts what it bought against the limit', async () => {
            const [open] = await pages();
            stripe.paid.add(open.session);
            const waiting = await buy(1000);
            expect(waiting.status).toBe(409);
            expect(waiting.data.code).toBe('PAYMENT_UNDER_WAY');

            await fromStripe(paid(person.id, 2000, 'pagepaid', { id: open.session }));
            expect(await pages()).toHaveLength(0);
            expect(await balance(person.id)).toBe(2000);
            const over = await buy(1000);
            expect(over.status).toBe(400);
            expect(over.data).toMatchObject({ code: 'TOO_LARGE', most: 500 });
        });

        it('lets the page go when Stripe will not make it, and forgets a page past its end', async () => {
            await makeOlder(person.id, 30);
            stripe.refuse = 'Not today';
            try {
                expect((await buy(1000)).status).toBe(502);
            } finally {
                stripe.refuse = null;
            }
            expect(await pages()).toHaveLength(0);
            await Checkout.create({ service: BILLING_SERVICE, live: false, user: person.id, session: 'cs_old', amount: 1000, endsAt: new Date(Date.now() - 1000) });
            const before = stripe.calls.length;
            expect((await buy(1000)).status).toBe(200);
            expect(stripe.calls.slice(before).map((call) => call.path)).toEqual(['/checkout/sessions']);
        });
    });

    describe('Disputes', () => {
        let person;
        const disputed = (name, cents, status, created = true) => ({
            id: `evt_${run}_${name}_${status}`,
            type: created ? 'charge.dispute.created' : 'charge.dispute.closed',
            data: { object: { id: `dp_${run}_${name}`, payment_intent: `pi_${run}_${name}`, charge: `ch_${run}_${name}`, amount: cents, status } }
        });

        beforeAll(async () => {
            person = await testStartup.createMutableUser({ role: 'USER', prefix: `dispute${run}` });
        });

        it('takes the credit back while the bank looks into it, once, and tells the person and the owners', async () => {
            await fromStripe(paid(person.id, 3000, 'contested'));
            const event = disputed('contested', 3000, 'needs_response');
            await Promise.all([fromStripe(event), fromStripe(event), fromStripe({ ...event, id: `${event.id}_again` })]);
            expect(await balance(person.id)).toBe(0);
            expect(await CreditLine.countDocuments({ service: BILLING_SERVICE, user: person.id, kind: 'dispute' })).toBe(1);
            await testStartup.loginAsUser('owner');
            await expect.poll(async () => (await client.get('/api/v1/notifications')).data.notifications.some((entry) => entry.type === 'billing:dispute'), { timeout: 5000 }).toBe(true);
        });

        it('gives it back when the dispute is won, once', async () => {
            const won = disputed('contested', 3000, 'won', false);
            await fromStripe(won);
            await fromStripe({ ...won, id: `${won.id}_again` });
            expect(await balance(person.id)).toBe(3000);
            await fromStripe(disputed('contested', 3000, 'lost', false));
            expect(await balance(person.id)).toBe(3000);
        });

        it('takes only what is unspent and not refunded, notes the rest, and gives nothing back when lost', async () => {
            await fromStripe(paid(person.id, 2000, 'spentfirst'));
            await billing.charge(person.id, 4000, { about: 'Used a card' });
            expect(await balance(person.id)).toBe(1000);
            await fromStripe(disputed('spentfirst', 2000, 'needs_response'));
            expect(await balance(person.id)).toBe(0);
            const line = await CreditLine.findOne({ service: BILLING_SERVICE, user: person.id, kind: 'dispute' }).sort({ seq: -1 });
            expect([line.amount, line.short]).toEqual([-1000, 1000]);
            await fromStripe(disputed('spentfirst', 2000, 'lost', false));
            expect(await balance(person.id)).toBe(0);
        });

        it('counts the purchase first when Stripe sends the dispute before the payment, and each only once', async () => {
            const buyer = await testStartup.createMutableUser({ role: 'USER', prefix: `early${run}` });
            const payment = paid(buyer.id, 1500, 'disputedfirst');
            stripe.sessions.set(payment.data.object.payment_intent, payment.data.object);
            await fromStripe(disputed('disputedfirst', 1500, 'needs_response'));
            await fromStripe(payment);
            const lines = await CreditLine.find({ service: BILLING_SERVICE, user: buyer.id }).sort({ seq: 1 });
            expect(lines.map((line) => [line.kind, line.amount])).toEqual([['topup', 1500], ['dispute', -1500]]);
            expect(await balance(buyer.id)).toBe(0);
        });

        it('ignores a dispute over a payment that is not this app\'s, or one that was never opened here', async () => {
            await fromStripe(disputed('somebody-elses-invoice', 9000, 'needs_response'));
            await fromStripe(disputed('never-opened', 1000, 'won', false));
            expect(await balance(person.id)).toBe(0);
            expect(await CreditLine.countDocuments({ service: BILLING_SERVICE, user: person.id, kind: 'dispute' })).toBe(3);
        });
    });

    describe('The owner\'s tools', () => {
        it('shows an administrator a person\'s ledger, and nobody else', async () => {
            await testStartup.loginAsUser('admin');
            const response = await client.get(`/api/v1/billing/people/${buyer.id}`);
            expect(response.status).toBe(200);
            expect(response.data.person).toMatchObject({ id: buyer.id, username: buyer.username });
            expect(response.data.balance).toBe(480);
            expect(response.data.lines.length).toBeGreaterThan(0);
            expect((await client.get(`/api/v1/billing/people/${new mongoose.Types.ObjectId()}`)).status).toBe(404);

            await testStartup.loginAsUser('user');
            expect((await client.get(`/api/v1/billing/people/${other.id}`)).status).toBe(403);
            expect((await client.post(`/api/v1/billing/people/${buyer.id}/adjustments`, { amount: 100000, reason: 'For me' })).status).toBe(403);
            expect(await balance(buyer.id)).toBe(480);
        });

        it('finds people by the start of a username or email, for administrators only', async () => {
            await testStartup.loginAsUser('admin');
            const found = await client.get(`/api/v1/billing/people?find=${buyer.username.slice(0, 8).toUpperCase()}`);
            expect(found.status).toBe(200);
            expect(found.data.people.find((person) => person.id === buyer.id)).toMatchObject({ username: buyer.username, balance: 480, held: 0 });
            expect((await client.get('/api/v1/billing/people?find=.*')).data.people).toEqual([]);
            expect((await client.get('/api/v1/billing/people?find=a')).data.people).toEqual([]);

            await testStartup.loginAsUser('user');
            expect((await client.get('/api/v1/billing/people?find=user')).status).toBe(403);
        });

        it('keeps test money and real money apart, though they may share a database', async () => {
            const key = process.env.STRIPE_SECRET_KEY;
            const person = testStartup.superCreator.id;
            const real = async (work) => {
                process.env.STRIPE_SECRET_KEY = 'rk_live_standin';
                try {
                    return await work();
                } finally {
                    process.env.STRIPE_SECRET_KEY = key;
                }
            };
            await fromStripe({ ...paid(person, 2000, 'testmoney'), livemode: false });
            expect(await balance(person)).toBe(2000);

            // Where money is real the test credit is not there, cannot be spent or refunded, and a test message adds none.
            await real(async () => {
                expect(await billing.balanceOf(person)).toEqual({ balance: 0, held: 0 });
                expect((await billing.linesOf(person)).lines).toEqual([]);
                await expect(billing.setAside(person, 500, 'apart', 'A card')).rejects.toMatchObject({ code: 'NOT_ENOUGH_CREDIT' });
                await fromStripe({ ...paid(person, 3000, 'testagain'), livemode: false });
                await fromStripe({ ...paid(person, 3000, 'unsaid') });
                expect(await balance(person)).toBe(0);
                await fromStripe({ ...paid(person, 1500, 'realmoney'), livemode: true });
                expect(await balance(person)).toBe(1500);
            });

            // And the other way round; line numbers run over both, so neither blocks the other.
            expect(await balance(person)).toBe(2000);
            await fromStripe({ ...paid(person, 4000, 'realhere'), livemode: true });
            await fromStripe({ ...refunded('realmoney', 1500, 'realrefund'), livemode: true });
            expect(await balance(person)).toBe(2000);
            await billing.charge(person, 500, { about: 'Spent' });
            expect(await balance(person)).toBe(1500);
            const lines = await CreditLine.find({ service: BILLING_SERVICE, user: person }).sort({ seq: 1 });
            expect(lines.map((line) => [line.seq, line.live, line.balance])).toEqual([[1, false, 2000], [2, true, 1500], [3, false, 1500]]);
            await real(async () => expect(await balance(person)).toBe(1500));
        });

        it('adds an adjustment with a reason, and never below zero', async () => {
            await testStartup.loginAsUser('admin');
            const noReason = await client.post(`/api/v1/billing/people/${buyer.id}/adjustments`, { amount: 500 });
            expect(noReason.status).toBe(400);
            expect(noReason.data.code).toBe('NO_REASON');
            for (const amount of [0, 1.5, 'lots', 100001]) {
                const bad = await client.post(`/api/v1/billing/people/${buyer.id}/adjustments`, { amount, reason: 'A reason' });
                expect(bad.data.code).toBe('BAD_AMOUNT');
            }

            const added = await client.post(`/api/v1/billing/people/${buyer.id}/adjustments`, { amount: 520, reason: 'Sorry for the outage' });
            expect(added.status).toBe(201);
            expect(added.data).toMatchObject({ balance: 1000, line: { kind: 'adjustment', change: 520, about: 'Sorry for the outage' } });

            const below = await client.post(`/api/v1/billing/people/${buyer.id}/adjustments`, { amount: -1001, reason: 'Too much' });
            expect(below.status).toBe(402);
            const taken = await client.post(`/api/v1/billing/people/${buyer.id}/adjustments`, { amount: -1000, reason: 'Taken back' });
            expect(taken.status).toBe(201);
            expect(await balance(buyer.id)).toBe(0);
        });
    });
});
